#!/usr/bin/env python3
"""
Synthetic CNV Data Generator with Sex-Specific Configurations, WGD Option, and PAR Handling
"""

import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm

class CNVSegment:
    """
    Represents a CNV segment with chromosome, start and end positions, and nMajor and nMinor copy numbers.
    """
    def __init__(self, chrom, startpos, endpos, nMajor, nMinor):
        self.chrom = chrom
        self.startpos = startpos
        self.endpos = endpos
        self.nMajor = nMajor
        self.nMinor = nMinor

    def to_list(self):
        return [self.chrom, self.startpos, self.endpos, self.nMajor, self.nMinor]

class CNVDataLoader:
    """
    Loads existing CNV data to extract distributions for nMajor, nMinor, segment lengths, and segments per chromosome.
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.nMajor_values = []
        self.nMinor_values = []
        self.segment_lengths = []
        self.segments_per_chromosome = []

    def load_data(self):
        """
        Loads CNV data from files in the specified folder and computes distributions.
        """
        for root, dirs, files in os.walk(self.folder_path):
            for filename in files:
                if filename.endswith(('.tsv', '.txt', '.csv')):
                    file_path = os.path.join(root, filename)
                    df = pd.read_csv(file_path, sep='\t', header=0)
                    required_columns = {'chr', 'startpos', 'endpos', 'nMajor', 'nMinor'}
                    if not required_columns.issubset(df.columns):
                        continue  # Skip files without required columns
                    df = df.dropna(subset=required_columns)
                    df['nMajor'] = df['nMajor'].astype(int)
                    df['nMinor'] = df['nMinor'].astype(int)
                    df['startpos'] = df['startpos'].astype(int)
                    df['endpos'] = df['endpos'].astype(int)

                    self.nMajor_values.extend(df['nMajor'].tolist())
                    self.nMinor_values.extend(df['nMinor'].tolist())
                    self.segment_lengths.extend((df['endpos'] - df['startpos']).tolist())
                    segments_per_chr = df.groupby('chr').size().tolist()
                    self.segments_per_chromosome.extend(segments_per_chr)

        # Create distributions
        nMajor_counts = pd.Series(self.nMajor_values).value_counts(normalize=True)
        nMinor_counts = pd.Series(self.nMinor_values).value_counts(normalize=True)

        self.nMajor_dist = nMajor_counts.sort_index()
        self.nMinor_dist = nMinor_counts.sort_index()

        return self.nMajor_dist, self.nMinor_dist

    def calculate_segment_length_distribution(self):
        """
        Returns the segment length distribution.
        """
        return self.segment_lengths

    def calculate_segments_per_chromosome(self):
        """
        Calculates the average number of segments per chromosome using a Poisson distribution.
        """
        lambda_value = np.mean(self.segments_per_chromosome)
        return lambda_value

class SyntheticCNVGenerator:
    """
    Generates synthetic CNV data by sampling from existing distributions and creating new CNV segments.
    Handles male/female differences through X chromosome copy numbers, applies WGD when enabled, and correctly handles PAR regions.
    """

    def __init__(self, nMajor_dist, nMinor_dist, segment_lengths, poisson_lambda, male_nonpar_gain_prob=0.05):
        self.nMajor_dist = nMajor_dist
        self.nMinor_dist = nMinor_dist
        self.segment_lengths = segment_lengths
        self.poisson_lambda = poisson_lambda
        self.male_nonpar_gain_prob = male_nonpar_gain_prob  # Probability of CN gain in male non-PAR regions
        self.chromosome_boundaries = {
            "1": [(61735, 248930189)],
            "2": [(12784, 242193529)],
            "3": [(18667, 198295559)],
            "4": [(12281, 190214555)],
            "5": [(15532, 181538259)],
            "6": [(149661, 170805979)],
            "7": [(43259, 159345973)],
            "8": [(81254, 145138636)],
            "9": [(46587, 138394717)],
            "10": [(26823, 133797422)],
            "11": [(198572, 135086622)],
            "12": [(51460, 133275309)],
            "13": [(18452809, 114364328)],
            "14": [(18225647, 107043718)],
            "15": [(19811075, 101991189)],
            "16": [(10777, 90338345)],
            "17": [(150733, 83257441)],
            "18": [(48133, 80373285)],
            "19": [(90910, 58617616)],
            "20": [(80664, 64444167)],
            "21": [(10336543, 46709983)],
            "22": [(15294545, 50818468)],
            "X": [
                # Pseudoautosomal Region 1 (PAR1)
                (60001, 2699520),
                # Non-PAR region between PAR1 and PAR2
                (2699521, 154931043),
                # Pseudoautosomal Region 2 (PAR2)
                (154931044, 155260560),
                # Remaining X chromosome end
                (155260561, 156040895)
            ]
        }

    def sample_nMajor_nMinor(self, chrom, sex, wgd=False, region=None):
        """
        Sample nMajor and nMinor values, with special handling for X chromosome based on sex and region.
        """
        if chrom == "X":
            if sex == "male":
                # Determine if the region is within PARs
                in_par = False
                if region:
                    start, end = region
                    # Check if the region overlaps with PARs
                    par_regions = [
                        (60001, 2699520),      # PAR1
                        (154931044, 155260560) # PAR2
                    ]
                    for par_start, par_end in par_regions:
                        if not (end < par_start or start > par_end):
                            in_par = True
                            break

                if in_par:
                    # Male PAR regions are diploid
                    nMajor = 1
                    nMinor = 1
                else:
                    # Male non-PAR regions
                    # Decide if a copy number gain occurs based on probability
                    if np.random.rand() < self.male_nonpar_gain_prob:
                        # Copy number gain: sample nMinor >=1
                        # To ensure nMinor >=1, we adjust the distribution
                        # Shift the probabilities to exclude nMinor=0
                        nMinor_options = self.nMinor_dist.index[self.nMinor_dist.index >=1]
                        nMinor_probs = self.nMinor_dist.loc[nMinor_options].values
                        nMinor_probs = nMinor_probs / nMinor_probs.sum()  # Re-normalize
                        nMinor = np.random.choice(nMinor_options, p=nMinor_probs)
                        nMajor = 1  # Typically remains haploid
                        # Apply WGD if enabled
                        if wgd:
                            nMajor = nMajor * 2
                            nMinor = nMinor * 2
                    else:
                        # No copy number gain
                        nMajor = 1
                        nMinor = 0
            else:
                # Female samples have normal diploid X chromosome
                nMajor = np.random.choice(self.nMajor_dist.index, p=self.nMajor_dist.values)
                nMinor = np.random.choice(self.nMinor_dist.index, p=self.nMinor_dist.values)
        else:
            # For autosomes sample normally
            nMajor = np.random.choice(self.nMajor_dist.index, p=self.nMajor_dist.values)
            nMinor = np.random.choice(self.nMinor_dist.index, p=self.nMinor_dist.values)

            if wgd:
                # Apply whole genome doubling, but maintain male X chromosome pattern
                nMajor = nMajor * 2
                nMinor = nMinor * 2

        # Ensure that nMajor and nMinor are integers
        nMajor = int(nMajor)
        nMinor = int(nMinor)

        return nMajor, nMinor

    def generate_segments_for_chromosome(self, chrom, regions, num_segments_list, sex, wgd=False):
        """
        Generate segments for a given chromosome with sex-specific handling of X chromosome.
        """
        segments = []

        for region_idx, (start_boundary, end_boundary) in enumerate(regions):
            num_segments = num_segments_list[region_idx]

            # If the region is a PAR region in males, force num_segments to 1
            if chrom == "X" and sex == "male":
                # Check if this region is a PAR region
                par_regions = [
                    (60001, 2699520),      # PAR1
                    (154931044, 155260560) # PAR2
                ]
                current_region = (start_boundary, end_boundary)
                is_par = False
                for par_start, par_end in par_regions:
                    if not (current_region[1] < par_start or current_region[0] > par_end):
                        is_par = True
                        break
                if is_par:
                    num_segments = 1  # Ensure single segment for PARs

            # Calculate segment lengths to fit within boundaries
            segment_lengths = []
            remaining_length = end_boundary - start_boundary

            for _ in range(num_segments - 1):
                if remaining_length <= 0:
                    break
                length = min(int(np.random.choice(self.segment_lengths)), remaining_length)
                if length > 0:
                    segment_lengths.append(length)
                    remaining_length -= length
                else:
                    break  # No more length to allocate

            # Add final segment with remaining length
            if remaining_length > 0:
                segment_lengths.append(remaining_length)

            # Generate segments based on the calculated lengths
            current_pos = start_boundary
            for length in segment_lengths:
                startpos = current_pos
                endpos = startpos + length

                # Sample nMajor and nMinor with sex-specific handling
                nMajor, nMinor = self.sample_nMajor_nMinor(
                    chrom, sex, wgd=wgd, region=(startpos, endpos)
                )

                segment = CNVSegment(chrom, startpos, endpos, nMajor, nMinor)
                segments.append(segment)
                current_pos = endpos

        return segments

    def generate_synthetic_sample(self, sample_id, sex='female', wgd=False):
        """
        Generate a synthetic sample with appropriate sex-specific CNV patterns.
        """
        if sex not in ['male', 'female']:
            raise ValueError("Sex must be either 'male' or 'female'")

        sample_segments = []
        for chrom, regions in tqdm(self.chromosome_boundaries.items(), desc=f"Generating segments for {sample_id}"):
            # Generate number of segments for each region
            num_segments_list = []
            for region_idx, _ in enumerate(regions):
                # For PAR regions in males, ensure only 1 segment
                if chrom == "X" and sex == "male":
                    par_regions = [
                        (60001, 2699520),      # PAR1
                        (154931044, 155260560) # PAR2
                    ]
                    current_region = regions[region_idx]
                    if any(not (current_region[1] < par_start or current_region[0] > par_end) for par_start, par_end in par_regions):
                        num_segments = 1  # Force single segment for PARs
                    else:
                        # Use Poisson distribution for non-PAR regions
                        num_segments = max(1, int(np.random.poisson(lam=self.poisson_lambda)))
                else:
                    # Use Poisson distribution for all other regions
                    num_segments = max(1, int(np.random.poisson(lam=self.poisson_lambda)))
                num_segments_list.append(num_segments)

            # Generate segments with sex-specific handling
            segments = self.generate_segments_for_chromosome(
                chrom, regions, num_segments_list, sex=sex, wgd=wgd
            )

            for segment in segments:
                sample_segments.append([sample_id] + segment.to_list())

        return sample_segments

    def save_synthetic_data(self, output_folder, num_samples=5, sex='female', wgd=False):
        """
        Save synthetic data as TSV files for the specified sex.

        Args:
            output_folder: Path to save output files
            num_samples: Total number of samples to generate
            sex: Sex of the samples ('male' or 'female')
            wgd: Whether to apply whole genome duplication
        """
        os.makedirs(output_folder, exist_ok=True)

        # Generate samples with the specified sex
        for i in tqdm(range(num_samples), desc="Generating synthetic samples"):
            sample_id = f"SYN-SAMPLE-{i:04d}"

            sample_data = self.generate_synthetic_sample(sample_id, sex=sex, wgd=wgd)

            # Save as TSV
            df = pd.DataFrame(sample_data, columns=["sample", "chr", "startpos", "endpos", "nMajor", "nMinor"])
            output_path = os.path.join(output_folder, f"{sample_id}_{sex}.tsv")
            df.to_csv(output_path, sep="\t", index=False)

            print(f"Generated synthetic {sex} data for {sample_id} at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CNV data with sex-specific configurations.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to existing CNV data folder for distribution analysis.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save the synthetic CNV samples.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of synthetic samples to generate.")
    parser.add_argument("--sex", type=str, choices=['male', 'female'], required=True, help="Sex of the synthetic samples.")
    parser.add_argument("--wgd", action='store_true', help="Enable Whole Genome Duplication (WGD) in synthetic samples.")
    args = parser.parse_args()

    # Load existing data to get distributions
    loader = CNVDataLoader(args.folder_path)
    nMajor_dist, nMinor_dist = loader.load_data()
    segment_lengths = loader.calculate_segment_length_distribution()
    poisson_lambda = loader.calculate_segments_per_chromosome()

    # Create generator and generate samples
    generator = SyntheticCNVGenerator(nMajor_dist, nMinor_dist, segment_lengths, poisson_lambda)
    generator.save_synthetic_data(
        args.output_folder,
        num_samples=args.num_samples,
        sex=args.sex,
        wgd=args.wgd
    )

if __name__ == "__main__":
    main()
