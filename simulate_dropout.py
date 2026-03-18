#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate modal dropout by renaming files based on given probabilities.

This script walks through the KRadar dataset structure, randomly decides
whether to 'drop' camera or radar modalities based on specified
probabilities, and renames the corresponding files to mark them as
unavailable (e.g., mono.jpg -> mono_unable.jpg).
It can target specific subsets: train, test, val.
"""

import os
import random
import argparse
from pathlib import Path


def simulate_dropout(root_dir, cam_dropout_prob, radar_dropout_prob, subsets, seed=None):
    """
    Simulates modal dropout by renaming files.

    Args:
        root_dir (str or Path): Path to the root of the KRadar dataset (e.g., 'data/kradar').
        cam_dropout_prob (float): Probability of dropping the camera modality (0.0 to 1.0).
        radar_dropout_prob (float): Probability of dropping the radar modality (0.0 to 1.0).
        subsets (list of str): List of subsets to process (e.g., ['train', 'test']).
        seed (int, optional): Seed for random number generator for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
        print(f"Set random seed to: {seed}")

    print(f"Starting simulation in directory: {root_dir}")
    print(f"Target subsets: {subsets}")
    print(f"Camera dropout probability: {cam_dropout_prob}")
    print(f"Radar dropout probability: {radar_dropout_prob}")

    # Define the subdirectories to process based on user input
    # subdirs_to_process = ['test', 'train', 'val']  # Assuming these exist under kradar/
    subdirs_to_process = [s.lower() for s in subsets] # Ensure lowercase matching

    for subdir_name in subdirs_to_process:
        subdir_path = Path(root_dir) / subdir_name
        if not subdir_path.exists():
            print(f"Warning: Subdirectory {subdir_path} does not exist, skipping.")
            continue

        print(f"Processing subdirectory: {subdir_path}")

        # Walk through all sequence directories (e.g., 1, 2, 3, ...)
        for sequence_dir in subdir_path.iterdir():
            if not sequence_dir.is_dir():
                continue

            # Walk through all timestamp directories within each sequence directory
            for timestamp_dir in sequence_dir.iterdir():
                if not timestamp_dir.is_dir():
                    continue

                print(f"  Processing timestamp directory: {timestamp_dir}")

                # --- Simulate Camera Dropout ---
                mono_path = timestamp_dir / "mono.jpg"
                mono_unable_path = timestamp_dir / "mono_unable.jpg"
                if mono_path.exists():
                    # Check if already renamed, to avoid conflicts if running script multiple times
                    if mono_unable_path.exists():
                        print(f"    Camera file already marked as unavailable: {mono_unable_path}")
                    else:
                        # Decide whether to drop the camera modality
                        if random.random() < cam_dropout_prob:
                            print(f"    DROPPING camera modality for {timestamp_dir.name}")
                            try:
                                mono_path.rename(mono_unable_path)
                                print(f"      Renamed: {mono_path.name} -> {mono_unable_path.name}")
                            except OSError as e:
                                print(f"      Error renaming {mono_path}: {e}")
                        else:
                            print(f"    Warning: Camera file {mono_path} not found in {timestamp_dir}")

                # --- Simulate Radar Dropout ---
                # Check both ra.npy and ea.npy, as they likely represent the radar data together
                ra_path = timestamp_dir / "ra.npy"
                ea_path = timestamp_dir / "ea.npy"
                ra_unable_path = timestamp_dir / "ra_unable.npy"
                ea_unable_path = timestamp_dir / "ea_unable.npy"

                # Check if both radar files exist
                ra_exists = ra_path.exists()
                ea_exists = ea_path.exists()
                # Check if already renamed
                ra_unable_exists = ra_unable_path.exists()
                ea_unable_exists = ea_unable_path.exists()

                if ra_unable_exists and ea_unable_exists:
                    print(f"    Radar files already marked as unavailable: {ra_unable_path}, {ea_unable_path}")
                elif ra_exists and ea_exists:
                    # Decide whether to drop the radar modality
                    if random.random() < radar_dropout_prob:
                        print(f"    DROPPING radar modality for {timestamp_dir.name}")
                        success_ra = True
                        success_ea = True
                        if ra_exists:
                            try:
                                ra_path.rename(ra_unable_path)
                                print(f"      Renamed: {ra_path.name} -> {ra_unable_path.name}")
                            except OSError as e:
                                print(f"      Error renaming {ra_path}: {e}")
                                success_ra = False
                        if ea_exists:
                            try:
                                ea_path.rename(ea_unable_path)
                                print(f"      Renamed: {ea_path.name} -> {ea_unable_path.name}")
                            except OSError as e:
                                print(f"      Error renaming {ea_path}: {e}")
                                success_ea = False
                        
                        if not success_ra or not success_ea:
                             print(f"      Error: Failed to rename some radar files in {timestamp_dir}. Please check manually.")
                    else:
                        missing_files = []
                        if not ra_exists and not ra_unable_exists:
                            missing_files.append(ra_path.name)
                        if not ea_exists and not ea_unable_exists:
                            missing_files.append(ea_path.name)
                        if missing_files:
                            print(f"    Warning: Radar file(s) missing in {timestamp_dir}: {', '.join(missing_files)}")

    print("Simulation completed.")


def restore_files(root_dir, subsets):
    """
    Helper function to restore renamed files back to their original names.

    Args:
        root_dir (str or Path): Path to the root of the KRadar dataset.
        subsets (list of str): List of subsets to process for restoration (e.g., ['train', 'test']).
    """
    print(f"Starting restoration in directory: {root_dir}")
    subdirs_to_process = [s.lower() for s in subsets] # Ensure lowercase matching

    for subdir_name in subdirs_to_process:
        subdir_path = Path(root_dir) / subdir_name
        if not subdir_path.exists():
            print(f"Warning: Subdirectory {subdir_path} does not exist, skipping.")
            continue

        print(f"Processing subdirectory for restoration: {subdir_path}")
        for sequence_dir in subdir_path.iterdir():
            if not sequence_dir.is_dir():
                continue
                
            # Walk through all timestamp directories within each sequence directory
            for timestamp_dir in sequence_dir.iterdir():
                if not timestamp_dir.is_dir():
                    continue
                    
                # Restore Camera
                mono_unable_path = timestamp_dir / "mono_unable.jpg"
                mono_path = timestamp_dir / "mono.jpg"
                if mono_unable_path.exists():
                    try:
                        mono_unable_path.rename(mono_path)
                        print(f"  Restored camera: {mono_unable_path.name} -> {mono_path.name}")
                    except OSError as e:
                        print(f"  Error restoring {mono_unable_path}: {e}")

                # Restore Radar
                ra_unable_path = timestamp_dir / "ra_unable.npy"
                ra_path = timestamp_dir / "ra.npy"
                if ra_unable_path.exists():
                    try:
                        ra_unable_path.rename(ra_path)
                        print(f"  Restored radar ra: {ra_unable_path.name} -> {ra_path.name}")
                    except OSError as e:
                        print(f"  Error restoring {ra_unable_path}: {e}")

                ea_unable_path = timestamp_dir / "ea_unable.npy"
                ea_path = timestamp_dir / "ea.npy"
                if ea_unable_path.exists():
                    try:
                        ea_unable_path.rename(ea_path)
                        print(f"  Restored radar ea: {ea_unable_path.name} -> {ea_path.name}")
                    except OSError as e:
                        print(f"  Error restoring {ea_unable_path}: {e}")

    print("Restoration completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate modal dropout in KRadar dataset by renaming files."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the root of the KRadar dataset (e.g., data/kradar)"
    )
    parser.add_argument(
        "--cam_dropout_prob",
        type=float,
        default=0.0,
        help="Probability of dropping the camera modality (default: 0.0)"
    )
    parser.add_argument(
        "--radar_dropout_prob",
        type=float,
        default=0.0,
        help="Probability of dropping the radar modality (default: 0.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for random number generator for reproducibility (default: None)"
    )
    parser.add_argument(
        "--restore",
        action='store_true',
        help="Restore previously renamed files back to their original names."
    )
    parser.add_argument(
        "--subsets",
        nargs='+',
        type=str,
        required=True, # Now required to specify subsets
        choices=['train', 'test', 'val'], # Restrict choices
        help="Specify which subsets to process (e.g., --subsets train test val). Choices: train, test, val"
    )

    args = parser.parse_args()

    if args.restore:
        restore_files(args.dataset_path, args.subsets)
    else:
        # Validate probabilities
        if not (0.0 <= args.cam_dropout_prob <= 1.0):
            parser.error("cam_dropout_prob must be between 0.0 and 1.0")
        if not (0.0 <= args.radar_dropout_prob <= 1.0):
            parser.error("radar_dropout_prob must be between 0.0 and 1.0")
        
        simulate_dropout(
            args.dataset_path,
            args.cam_dropout_prob,
            args.radar_dropout_prob,
            args.subsets,
            seed=args.seed
        )