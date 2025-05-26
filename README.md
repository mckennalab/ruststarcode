# ruststarcode

A Rust implementation of the StarCode algorithm for efficient sequence clustering and barcode collapsing. This library provides fast, memory-efficient clustering of DNA/RNA sequences with configurable edit distance thresholds.

## Overview

StarCode is an algorithm designed to cluster sequences (particularly barcodes or short DNA sequences) based on edit distance. This Rust implementation uses an optimized trie data structure with partial dynamic programming matrices to efficiently find similar sequences within a specified edit distance threshold.

### Key Features

- **Efficient Trie-based Storage**: Sequences sharing common prefixes are stored efficiently
- **Partial Needleman-Wunsch**: Optimized dynamic programming for edit distance calculation
- **Message Passing Clustering**: Advanced clustering algorithm that collapses low-frequency sequences into high-frequency ones
- **Memory Optimized**: Uses custom allocator (jemalloc) for better memory performance
- **Thread-safe**: Designed with shared ownership patterns using `Rc<RefCell<T>>`

## Algorithm Details

### 1. Trie Construction
Sequences are inserted into a trie where each node represents a prefix. Each node maintains:
- Partial dynamic programming matrix for alignment
- Parent/child relationships
- Terminal status for complete sequences

### 2. Sequence Search
The `chained_search` method efficiently finds sequences within edit distance by:
- Leveraging shared prefixes to avoid redundant calculations
- Using partial DP matrices from parent nodes
- Pruning search space based on edit distance thresholds

### 3. Clustering via Message Passing
The clustering algorithm:
1. Builds a distance graph connecting similar sequences
2. Uses message passing to collapse low-count sequences into high-count ones
3. Applies a minimum ratio threshold to prevent over-clustering

## Usage

### Basic Example

```rust
use rust_star::LinkedDistances;

// Prepare sequence data: (sequence, count) pairs
let sequences = vec![
    (b"ACGTACGT".to_vec(), 100),
    (b"ACGTACGA".to_vec(), 1),    // 1 mismatch from first
    (b"ACGTACGC".to_vec(), 1),    // 1 mismatch from first
    (b"TTTTAAAA".to_vec(), 50),
];

// Cluster sequences with edit distance ≤ 1 and minimum ratio 5.0
let max_length = 8;
let max_edit_distance = 1;
let min_ratio = 5.0;

let clusters = LinkedDistances::cluster_string_vector_list(
    &max_length,
    sequences,
    &max_edit_distance,
    &min_ratio
);

// Process results
for (seq, node) in clusters {
    if node.borrow().valid {
        println!("Cluster representative: {} (count: {})", 
                 String::from_utf8(seq).unwrap(), 
                 node.borrow().count);
        
        // Show absorbed sequences
        for (absorbed_seq, absorbed_count) in &node.borrow().swallowed_links {
            println!("  └─ {}: {}", 
                     String::from_utf8(absorbed_seq.clone()).unwrap(), 
                     absorbed_count);
        }
    }
}
```

### Trie Operations

```rust
use rust_star::Trie;
use std::collections::HashSet;

let mut trie = Trie::new(10); // max sequence length = 10
let max_mismatches = 2;

// Insert sequences
trie.insert(b"ACGTACGT", None, &max_mismatches);
trie.insert(b"ACGTACGA", None, &max_mismatches);

// Search for similar sequences
let search_nodes = HashSet::new();
let (matches, _) = trie.chained_search(
    0,                    // start depth
    None,                 // future depth
    b"ACGTACGC",         // search sequence
    &max_mismatches,
    &search_nodes
);

for (seq, distance) in matches {
    println!("Found: {} (distance: {})", 
             String::from_utf8(seq).unwrap(), 
             distance);
}
```

## API Reference

### Core Types

#### `Trie`
Main trie data structure for sequence storage and search.

**Methods:**
- `new(max_height: usize) -> Self` - Create new trie
- `insert(&mut self, seq: &[u8], depth_to_return: Option<usize>, max_mismatch: &usize) -> Vec<Link<TrieNode>>` - Insert sequence
- `chained_search(&mut self, start_depth: usize, future_depth: Option<usize>, sequence: &[u8], max_mismatches: &usize, search_nodes: &HashSet<Link<TrieNode>>) -> (Vec<(Vec<u8>, usize)>, HashSet<Link<TrieNode>>)` - Search for similar sequences

#### `LinkedDistances`
Graph structure for sequence clustering.

**Methods:**
- `cluster_string_vector_list(max_string_length: &usize, strings: Vec<(Vec<u8>, usize)>, max_mismatch: &usize, minimum_ratio: &f64) -> Vec<(Vec<u8>, Link<DistanceGraphNode>)>` - Main clustering function

#### `TrieNode`
Individual node in the trie structure.

**Fields:**
- `sequence: Vec<u8>` - Sequence represented by this node
- `is_terminal: bool` - Whether this represents a complete sequence
- `children: HashMap<u8, Rc<RefCell<TrieNode>>>` - Child nodes

#### `PartialNW`
Partial Needleman-Wunsch alignment data.

**Fields:**
- `column: Vec<usize>` - Column values in DP matrix
- `row: Vec<usize>` - Row values in DP matrix  
- `best_value: usize` - Best alignment score

## Performance Considerations

- **Memory**: Uses jemalloc for improved allocation performance
- **Time Complexity**: O(n*m*k) where n=number of sequences, m=sequence length, k=edit distance
- **Space Complexity**: O(n*m) for trie storage plus O(k²) per node for DP matrices

## Testing

Run the test suite:

```bash
cargo test
```

For performance testing with release optimizations:

```bash
cargo test --release
```

## Contributing

When contributing to this codebase:
1. Ensure all tests pass
2. Add tests for new functionality
3. Follow existing code style and documentation patterns
4. Consider performance implications of changes

## License

See LICENSE file for licensing information.
