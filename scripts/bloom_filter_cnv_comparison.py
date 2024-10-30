import os
import pandas as pd
import math
import mmh3  # MurmurHash3 hashing
from bitarray import bitarray
import matplotlib.pyplot as plt
import argparse

class BloomFilter:
    def __init__(self, expected_items, false_positive_rate):
        """
        Initializes the Bloom filter.
        :param expected_items: Number of items expected to be stored in the filter
        :param false_positive_rate: Desired false positive rate
        """
        # Calculate the size of the bit array (m) and the number of hash functions (k)
        self.size = self._get_size(expected_items, false_positive_rate)
        self.hash_count = self._get_hash_count(self.size, expected_items)
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)
        print(f"Bloom filter initialized with size {self.size} and {self.hash_count} hash functions.")

    def add(self, item):
        """
        Adds an item to the Bloom filter.
        :param item: The item to add
        """
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            self.bit_array[digest] = True

    def __contains__(self, item):
        """
        Checks if an item is in the Bloom filter.
        :param item: The item to check
        :return: True if the item might be in the filter, False if the item is definitely not in the filter
        """
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            if not self.bit_array[digest]:
                return False
        return True

    @staticmethod
    def _get_size(n, p):
        """
        Calculates the size of the bit array (m) given expected number of items (n) and false positive rate (p).
        :param n: Expected number of items
        :param p: Desired false positive rate
        :return: Size of bit array
        """
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    @staticmethod
    def _get_hash_count(m, n):
        """
        Calculates the optimal number of hash functions (k) given the size of bit array (m) and number of items (n).
        :param m: Size of bit array
        :param n: Number of items
        :return: Number of hash functions
        """
        k = (m / n) * math.log(2)
        return max(1, int(k))  # Ensure at least one hash function

def read_segments(file_path):
    """
    Reads a copy number segment file into a DataFrame.
    """
    try:
        df = pd.read_csv(file_path, sep='\t', header=0)
        # Ensure required columns are present
        required_columns = {'chr', 'startpos', 'endpos', 'nMajor', 'nMinor'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"File {file_path} is missing required columns.")
        # Drop rows with missing data
        df = df.dropna(subset=required_columns)
        # Convert positions and copy numbers to appropriate types
        df['chr'] = df['chr'].astype(str)
        df['startpos'] = df['startpos'].astype(int)
        df['endpos'] = df['endpos'].astype(int)
        df['nMajor'] = df['nMajor'].astype(int)
        df['nMinor'] = df['nMinor'].astype(int)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame(columns=['chr', 'startpos', 'endpos', 'nMajor', 'nMinor'])

def segment_to_string_binned(row, bin_size):
    """
    Converts a segment into a binned string representation including copy number states.
    :param row: DataFrame row containing 'chr', 'startpos', 'endpos', 'nMajor', 'nMinor'
    :param bin_size: Size of the bins in base pairs
    :return: String representing the binned segment with copy number
    """
    start_bin = (row['startpos'] // bin_size) * bin_size
    end_bin = (row['endpos'] // bin_size) * bin_size
    return f"chr{row['chr']}_{start_bin}_{end_bin}_nMajor{row['nMajor']}_nMinor{row['nMinor']}"

def count_total_segments(true_samples_folder):
    """
    Counts the total number of segments across all true sample files.
    """
    total_segments = 0
    for root, dirs, files in os.walk(true_samples_folder):
        for filename in files:
            if filename.endswith(('.tsv', '.txt', '.csv')):
                file_path = os.path.join(root, filename)
                true_df = read_segments(file_path)
                total_segments += len(true_df)
    return total_segments

def plot_results(synthetic_df, bin_size, output_filename):
    """
    Plots the synthetic segments, highlighting matches, and saves the plot to a file.
    """
    # Convert chromosome identifiers to integers for sorting (handle 'X', 'Y', 'M')
    chrom_mapping = {str(i): i for i in range(1, 23)}
    chrom_mapping.update({'X': 23, 'Y': 24, 'M': 25, 'MT': 25})
    synthetic_df['Chromosome'] = synthetic_df['chr'].map(chrom_mapping)
    synthetic_df = synthetic_df.dropna(subset=['Chromosome'])
    synthetic_df['Chromosome'] = synthetic_df['Chromosome'].astype(int)

    # Sort the DataFrame
    synthetic_df = synthetic_df.sort_values(['Chromosome', 'startpos'])

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each segment
    for idx, row in synthetic_df.iterrows():
        chrom = row['Chromosome']
        start = row['startpos']
        end = row['endpos']
        match = row['Match']
        color = 'green' if match else 'red'
        ax.plot([start, end], [chrom, chrom], color=color, linewidth=5)

    # Customize the plot
    ax.set_xlabel('Genomic Position (bp)')
    ax.set_ylabel('Chromosome')
    ax.set_yticks(range(1, 25))
    ax.set_yticklabels([str(i) for i in range(1, 23)] + ['X', 'Y'])
    ax.set_title('Synthetic Sample Segments Comparison with Copy Number (Binned Analysis)')
    ax.grid(True)

    # Create a legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=4, label='Match'),
        Line2D([0], [0], color='red', lw=4, label='No Match')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Compare synthetic copy number segments with true samples using a Bloom filter.")
    parser.add_argument("--synthetic_sample_file", type=str, required=True, help="Path to the synthetic sample file.")
    parser.add_argument("--true_samples_folder", type=str, required=True, help="Path to the folder containing true sample files.")
    parser.add_argument("--bin_size", type=int, default=1000000, help="Bin size in base pairs (default: 1,000,000).")
    parser.add_argument("--output_filename", type=str, required=True, help="Filename to save the plot.")
    parser.add_argument("--false_positive_rate", type=float, default=0.001, help="Desired false positive rate for the Bloom filter (default: 0.1%).")
    args = parser.parse_args()

    # Read synthetic sample segments
    print("Reading synthetic sample segments...")
    synthetic_df = read_segments(args.synthetic_sample_file)
    total_synthetic_segments = len(synthetic_df)
    print(f"Total segments in synthetic sample: {total_synthetic_segments}")

    # Count total segments in true samples
    print("Counting total segments in true samples...")
    total_true_segments = count_total_segments(args.true_samples_folder)
    print(f"Total segments in true samples: {total_true_segments}")

    # Initialize custom Bloom filter with dynamic expected_num_segments
    expected_num_segments = total_true_segments
    bloom_filter = BloomFilter(expected_items=expected_num_segments, false_positive_rate=args.false_positive_rate)

    # Add true sample segments to the Bloom filter using binned positions and copy number states
    print("Adding true sample segments to the Bloom filter...")
    for root, dirs, files in os.walk(args.true_samples_folder):
        for filename in files:
            if filename.endswith(('.tsv', '.txt', '.csv')):
                file_path = os.path.join(root, filename)
                true_df = read_segments(file_path)
                for _, row in true_df.iterrows():
                    segment_str = segment_to_string_binned(row, args.bin_size)
                    bloom_filter.add(segment_str)
    print("Finished adding segments to the Bloom filter.")

    # Check synthetic sample segments against the Bloom filter and record matches
    print("Checking synthetic sample segments against the Bloom filter...")
    matches = 0
    match_results = []
    for _, row in synthetic_df.iterrows():
        segment_str = segment_to_string_binned(row, args.bin_size)
        if segment_str in bloom_filter:
            matches += 1
            match_results.append(True)
        else:
            match_results.append(False)

    # Add match results to synthetic_df
    synthetic_df['Match'] = match_results

    # Report results
    print(f"Number of matching segments: {matches}")
    if total_synthetic_segments > 0:
        percentage = (matches / total_synthetic_segments) * 100
    else:
        percentage = 0
    print(f"Percentage of synthetic segments found in true samples: {percentage:.2f}%")

    # Plot the results
    print("Plotting the results...")
    plot_results(synthetic_df, args.bin_size, args.output_filename)

if __name__ == "__main__":
    main()
