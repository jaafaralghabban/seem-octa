"""
SEEM-OCTA Trainer

This module provides a unified trainer class for training SEEM models
with either GIIR or Random click selection strategies.
"""

import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple

from ..core.lora import inject_lora, freeze_base_model, count_parameters, save_merged_model
from ..core.losses import CombinedLoss
from ..core.metrics import compute_metrics
from ..core.model_utils import find_best_matching_mask
from ..strategies.giir import get_giir_click_train_robust
from ..strategies.random_clicks import get_random_click
from ..visualization.plotting import plot_training_curves, plot_test_predictions
from .config import TrainingConfig

# SEEM imports
try:
    from modeling import build_model
    from utils.arguments import load_opt_from_config_files
except ImportError:
    print("Warning: SEEM modules not found")


class SEEMOCTATrainer:
    """
    Unified trainer for SEEM-OCTA models.
    
    Supports both GIIR and Random click selection strategies
    with LoRA-based parameter-efficient fine-tuning.
    
    Args:
        config: Training configuration
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device
        
        # Initialize model
        self.model = self._build_model()
        
        # Apply LoRA and freeze
        self._setup_lora()
        
        # Loss and optimizer
        self.criterion = CombinedLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=config.scheduler_patience,
            factor=config.scheduler_factor
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Training history
        self.history = defaultdict(list)
        self.best_dice = 0.0
    
    def _build_model(self) -> Any:
        """Build and load SEEM model."""
        print("Loading SEEM model...")
        opt = load_opt_from_config_files([self.config.config_path])
        model = build_model(opt)
        
        # Load pretrained weights
        if os.path.exists(self.config.pretrained_path):
            checkpoint = torch.load(self.config.pretrained_path, map_location='cpu')
            model.load_state_dict(
                checkpoint.get('model', checkpoint),
                strict=False
            )
        else:
            raise FileNotFoundError(
                f"Pretrained weights not found: {self.config.pretrained_path}"
            )
        
        model.to(self.device)
        print("✓ Model loaded")
        return model
    
    def _setup_lora(self):
        """Apply LoRA adapters and freeze base model."""
        print(f"Applying LoRA (rank={self.config.lora_rank}, alpha={self.config.lora_alpha})")
        
        # Inject LoRA into backbone
        inject_lora(
            self.model.backbone,
            r=self.config.lora_rank,
            alpha=self.config.lora_alpha
        )
        
        # Freeze base, unfreeze LoRA and decoder
        freeze_base_model(self.model)
        
        # Count parameters
        total, trainable, pct = count_parameters(self.model)
        print(f"✓ Parameters: {total:,} total, {trainable:,} trainable ({pct:.2f}%)")
        
        # Initialize text embeddings
        with torch.no_grad():
            self.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                ["vessel"], is_eval=False
            )
    
    def train_epoch_giir(
        self,
        dataloader: DataLoader,
        epoch_num: int
    ) -> float:
        """
        Train one epoch with GIIR strategy.
        
        GIIR: Geometry-Informed Interactive Refinement
        - Starts from zero clicks
        - Uses distance transform for optimal click placement
        - Topology-aware for thin vessels
        """
        self.model.train()
        total_loss = 0.0
        total_clicks = 0
        total_samples = 0
        
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Training GIIR (Epoch {epoch_num+1})"
        )
        
        for batch_idx, batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.config.mixed_precision):
                # Extract features once
                features = self.model.backbone(images)
                mask_feat, _, ms_feat = self.model.sem_seg_head.pixel_decoder.forward_features(features)
                
                batch_loss = torch.tensor(0.0, device=self.device)
                
                for i in range(images.size(0)):
                    s_mask = masks[i].unsqueeze(0)
                    s_ms_feat = [f[i].unsqueeze(0) for f in ms_feat]
                    s_mask_feat = mask_feat[i].unsqueeze(0)
                    
                    # Step 1: Get initial prediction (no clicks)
                    with torch.no_grad():
                        preds_0 = self.model.sem_seg_head.predictor(
                            s_ms_feat, s_mask_feat, extra={}, task='panoptic'
                        )
                        logits_0 = F.interpolate(
                            preds_0['pred_masks'],
                            size=s_mask.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                        best_logits_0 = find_best_matching_mask(logits_0, s_mask)
                        pred_mask_0 = (torch.sigmoid(best_logits_0) > 0.5).float()
                        
                        # Step 2: Find correction click using GIIR
                        correction_point, correction_label = get_giir_click_train_robust(
                            pred_mask_0, s_mask
                        )
                    
                    # Step 3: Train with correction
                    if correction_point is not None:
                        correction_point = correction_point.to(self.device)
                        correction_label = correction_label.to(self.device)
                        
                        prompts = {
                            "point_coords": correction_point.unsqueeze(0),
                            "point_labels": correction_label.unsqueeze(0)
                        }
                        
                        preds = self.model.sem_seg_head.predictor(
                            s_ms_feat, s_mask_feat, extra=prompts, task='panoptic'
                        )
                        all_logits = F.interpolate(
                            preds['pred_masks'],
                            size=s_mask.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                        best_logits = find_best_matching_mask(all_logits, s_mask)
                        loss = self.criterion(best_logits, s_mask)
                        batch_loss += loss
                        total_clicks += 1
                    else:
                        # Model was perfect, train on empty prompt
                        preds = self.model.sem_seg_head.predictor(
                            s_ms_feat, s_mask_feat, extra={}, task='panoptic'
                        )
                        logits = F.interpolate(
                            preds['pred_masks'],
                            size=s_mask.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                        best_logits = find_best_matching_mask(logits, s_mask)
                        loss = self.criterion(best_logits, s_mask)
                        batch_loss += loss
                    
                    total_samples += 1
            
            # Backward pass
            if batch_loss > 0:
                if self.scaler:
                    self.scaler.scale(batch_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    batch_loss.backward()
                    self.optimizer.step()
                
                total_loss += batch_loss.item()
                pbar.set_postfix({
                    'loss': f'{batch_loss.item() / images.size(0):.4f}',
                    'clicks': f'{total_clicks}/{total_samples}'
                })
        
        return total_loss / len(dataloader.dataset)
    
    def train_epoch_random(
        self,
        dataloader: DataLoader,
        epoch_num: int
    ) -> float:
        """
        Train one epoch with Random click strategy.
        
        Matches SEEM baseline inference behavior:
        - Starts from zero clicks
        - Random sampling from error regions
        - Accumulates clicks up to max_clicks
        """
        self.model.train()
        total_loss = 0.0
        total_clicks = 0
        total_samples = 0
        
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Training Random (Epoch {epoch_num+1})"
        )
        
        for batch_idx, batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.config.mixed_precision):
                features = self.model.backbone(images)
                mask_feat, _, ms_feat = self.model.sem_seg_head.pixel_decoder.forward_features(features)
                
                batch_loss = torch.tensor(0.0, device=self.device)
                
                for i in range(images.size(0)):
                    # Seeding for reproducibility
                    num_samples_so_far = batch_idx * dataloader.batch_size + i
                    seed = epoch_num * len(dataloader.dataset) + num_samples_so_far
                    
                    s_mask = masks[i].unsqueeze(0)
                    s_ms_feat = [f[i].unsqueeze(0) for f in ms_feat]
                    s_mask_feat = mask_feat[i].unsqueeze(0)
                    
                    # Start with empty clicks
                    all_points = torch.zeros((0, 2), device=self.device)
                    all_labels = torch.zeros((0,), dtype=torch.int64, device=self.device)
                    
                    sample_loss = torch.tensor(0.0, device=self.device)
                    clicks_this_sample = 0
                    
                    # Interactive loop
                    for click_idx in range(self.config.max_clicks_train):
                        # Forward pass
                        if all_points.size(0) > 0:
                            prompts = {
                                "point_coords": all_points.unsqueeze(0),
                                "point_labels": all_labels.unsqueeze(0)
                            }
                        else:
                            prompts = {}
                        
                        preds = self.model.sem_seg_head.predictor(
                            s_ms_feat, s_mask_feat, extra=prompts, task='panoptic'
                        )
                        all_logits = F.interpolate(
                            preds['pred_masks'],
                            size=s_mask.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                        best_logits = find_best_matching_mask(all_logits, s_mask)
                        
                        # Accumulate loss
                        step_loss = self.criterion(best_logits, s_mask)
                        sample_loss += step_loss
                        clicks_this_sample += 1
                        
                        # Generate next click
                        with torch.no_grad():
                            pred_np = (torch.sigmoid(best_logits) > 0.5).float().squeeze().cpu().numpy()
                            gt_np = s_mask.squeeze().cpu().numpy()
                            
                            history = [
                                (all_points[j, 0].item(), all_points[j, 1].item())
                                for j in range(all_points.size(0))
                            ]
                            
                            next_click = get_random_click(
                                pred_np, gt_np, history,
                                iteration_number=click_idx,
                                seed=seed
                            )
                        
                        if next_click is None:
                            break
                        
                        # Add click to history
                        new_pt = torch.tensor(
                            [[next_click['pt'][0], next_click['pt'][1]]],
                            dtype=torch.float32,
                            device=self.device
                        )
                        new_lbl = torch.tensor(
                            [next_click['lbl']],
                            dtype=torch.int64,
                            device=self.device
                        )
                        all_points = torch.cat([all_points, new_pt], dim=0)
                        all_labels = torch.cat([all_labels, new_lbl], dim=0)
                    
                    # Normalize loss
                    if clicks_this_sample > 0:
                        batch_loss += (sample_loss / clicks_this_sample)
                        total_clicks += clicks_this_sample
                        total_samples += 1
            
            # Backward pass
            if batch_loss > 0:
                if self.scaler:
                    self.scaler.scale(batch_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    batch_loss.backward()
                    self.optimizer.step()
                
                total_loss += batch_loss.item()
                avg_clicks = total_clicks / max(total_samples, 1)
                pbar.set_postfix({
                    'loss': f'{batch_loss.item() / images.size(0):.4f}',
                    'avg_clicks': f'{avg_clicks:.1f}'
                })
        
        return total_loss / len(dataloader.dataset)
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> Tuple[float, float, float, float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (loss, dice, iou, sensitivity, specificity)
        """
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_se = 0.0
        total_sp = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                points = batch['points'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with autocast(enabled=self.config.mixed_precision):
                    features = self.model.backbone(images)
                    mask_feat, _, ms_feat = self.model.sem_seg_head.pixel_decoder.forward_features(features)
                    
                    for i in range(images.size(0)):
                        s_mask = masks[i].unsqueeze(0)
                        s_ms_feat = [f[i].unsqueeze(0) for f in ms_feat]
                        s_mask_feat = mask_feat[i].unsqueeze(0)
                        
                        # Get valid points
                        valid_mask = labels[i] != -1
                        s_pts = points[i][valid_mask].unsqueeze(0)
                        s_lbls = labels[i][valid_mask].unsqueeze(0)
                        
                        if s_pts.size(1) == 0:
                            continue
                        
                        prompts = {
                            "point_coords": s_pts,
                            "point_labels": s_lbls
                        }
                        preds = self.model.sem_seg_head.predictor(
                            s_ms_feat, s_mask_feat, extra=prompts, task='panoptic'
                        )
                        all_logits = F.interpolate(
                            preds['pred_masks'],
                            size=s_mask.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                        logits = find_best_matching_mask(all_logits, s_mask)
                        
                        total_loss += self.criterion(logits, s_mask).item()
                        pred_binary = (torch.sigmoid(logits) > 0.5).float()
                        
                        dice, iou, se, sp = compute_metrics(pred_binary, s_mask)
                        total_dice += dice
                        total_iou += iou
                        total_se += se
                        total_sp += sp
                        count += 1
        
        if count == 0:
            return 0, 0, 0, 0, 0
        
        return (
            total_loss / count,
            total_dice / count,
            total_iou / count,
            total_se / count,
            total_sp / count
        )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            
        Returns:
            Training history dictionary
        """
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        # Select training function based on strategy
        if self.config.strategy == 'giir':
            train_fn = self.train_epoch_giir
        else:
            train_fn = self.train_epoch_random
        
        print(f"\nStarting training with {self.config.strategy.upper()} strategy")
        print(f"Epochs: {self.config.num_epochs}, Batch size: {self.config.batch_size}")
        print("=" * 70)
        
        for epoch in range(self.config.num_epochs):
            print(f"\n--- Epoch {epoch+1}/{self.config.num_epochs} ---")
            
            # Train
            train_loss = train_fn(train_loader, epoch)
            
            # Validate
            val_loss, val_dice, val_iou, val_se, val_sp = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_dice)
            self.history['val_iou'].append(val_iou)
            self.history['val_se'].append(val_se)
            self.history['val_sp'].append(val_sp)
            
            # Scheduler step
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_dice)
            
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, "
                  f"Val Dice={val_dice:.4f}, Val IoU={val_iou:.4f}, LR={current_lr:.1e}")
            
            # Save best model
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                best_path = os.path.join(
                    self.config.checkpoint_dir,
                    self.config.checkpoint_name
                )
                torch.save(self.model.state_dict(), best_path)
                print(f"  *** NEW BEST MODEL! Dice: {self.best_dice:.4f} ***")
        
        # Save final model
        final_path = os.path.join(
            self.config.checkpoint_dir,
            f"final_{self.config.checkpoint_name}"
        )
        torch.save(self.model.state_dict(), final_path)
        
        print("\n" + "=" * 70)
        print(f"TRAINING COMPLETE! Best Dice: {self.best_dice:.4f}")
        print("=" * 70)
        
        # Load best model for evaluation
        best_path = os.path.join(
            self.config.checkpoint_dir,
            self.config.checkpoint_name
        )
        self.model.load_state_dict(torch.load(best_path))
        
        # Plot training curves
        plot_training_curves(
            self.history,
            os.path.join(
                self.config.results_dir,
                f"training_curves_{self.config.strategy}.png"
            )
        )
        
        # Test evaluation
        if test_loader is not None:
            print("\nRunning test evaluation...")
            test_loss, test_dice, test_iou, test_se, test_sp = self.validate(test_loader)
            print(f"TEST RESULTS: Dice={test_dice:.4f}, IoU={test_iou:.4f}, "
                  f"SE={test_se:.4f}, SP={test_sp:.4f}")
        
        # Save merged model
        merged_path = os.path.join(
            self.config.checkpoint_dir,
            self.config.merged_checkpoint_name
        )
        save_merged_model(self.model, merged_path, self.device)
        
        return dict(self.history)
