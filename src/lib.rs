use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use std::rc::Rc;
use std::rc::Weak;
use std::cell::RefCell;

use std::fmt;
use std::fmt::Debug;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::ops::Deref;
use log::{debug};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

/// A wrapper around an `Rc<RefCell<T>>` that implements additional traits.
/// 
/// This structure provides a convenient way to share ownership of a mutable value
/// while maintaining reference semantics for hashing and equality comparison.
/// 
/// # Type Parameters
/// 
/// * `T` - The wrapped type, which must implement `Hash`, `PartialEq`, `Eq`, and `Debug`.
pub struct Link<T: Hash + PartialEq + Eq + Debug>(Rc<RefCell<T>>);


impl<T: Hash + PartialEq + Eq + Debug> Deref for Link<T> {
    type Target = Rc<RefCell<T>>;
    
    /// Dereferences to the inner `Rc<RefCell<T>>`.
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Hash + PartialEq + Eq + Debug> Hash for Link<T> {
    /// Hashes the inner value by borrowing it.
    /// 
    /// This implementation ensures that the hash is based on the contained value,
    /// not on the memory address of the `Rc`.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.borrow().hash(state);
    }
}

impl<T: Hash + PartialEq + Eq + Debug> Clone for Link<T> {
    /// Creates a new `Link` that shares ownership of the same inner value.
    fn clone(&self) -> Self {
        Link(self.0.clone())
    }
}

impl<T: Hash + PartialEq + Eq + Debug> Link<T> {
    /// Creates a new `Link` containing the provided value.
    ///
    /// # Parameters
    ///
    /// * `inner` - The value to be wrapped in the `Link`.
    pub fn new(inner: T) -> Self {
        Link(Rc::new(RefCell::new(inner)))
    }
}

impl<T: Hash + PartialEq + Eq + Debug> fmt::Debug for Link<T> {
    /// Formats the `Link` for debugging by delegating to the inner `Rc<RefCell<T>>`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: Hash + PartialEq + Eq + Debug> Eq for Link<T> {}


impl<T: Hash + PartialEq + Eq + Debug> PartialEq for Link<T> {
    /// Compares two `Link`s by comparing their inner `Rc`s.
    ///
    /// This implementation uses pointer equality to determine if two `Link`s refer to
    /// the same allocation.
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

/// A node in a trie data structure, optimized for sequence alignment.
///
/// This implementation includes functionality for partial Needleman-Wunsch dynamic
/// programming matrix calculations to efficiently find similar sequences.
#[derive(Debug)]
pub struct TrieNode {
    /// Children keyed by character. For DNA you might prefer a 4-element array (A,C,G,T),
    /// but using a HashMap<char, TrieNode> is more general.
    children: HashMap<u8, Rc<RefCell<TrieNode>>>,

    /// Reference to the parent node, allowing traversal up the trie.
    parent: Option<Weak<RefCell<TrieNode>>>,

    /// Tracks if this node has been visited in the current traversal.
    /// Used to avoid processing the same node multiple times.
    visited: usize,

    /// The sequence represented by the path from the root to this node.
    pub sequence: Vec<u8>,

    /// Indicates whether this node represents the end of a complete sequence.
    pub is_terminal: bool,

    /// Partial DP matrix for Needleman-Wunsch alignment algorithm.
    /// Stores a column and row fragment as described in the Starcode approach.
    partial_dp: PartialNW,

    /// The depth of this node in the trie (distance from root).
    depth: usize,
}

impl Hash for TrieNode {
    /// Hashes the node based on its sequence.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.sequence.hash(state);
    }
}

impl PartialEq for TrieNode {
    /// Compares nodes based on their sequences.
    fn eq(&self, other: &Self) -> bool {
        self.sequence == other.sequence
    }
}

impl Eq for TrieNode {}

/// Partial Needleman-Wunsch alignment data structure.
///
/// This struct holds the partial dynamic programming data that
/// helps skip repeated computations for sequences sharing a prefix.
/// Columns represent the search sequence, rows represent the known sequence.
#[derive(Debug, Clone)]
pub struct PartialNW {
    /// Column values in the dynamic programming matrix (contains the corner element).
    column: Vec<usize>, 
    
    /// Row values in the dynamic programming matrix.
    row: Vec<usize>,
    
    /// The best alignment score found so far.
    best_value: usize,
}

impl PartialNW {
    /// Creates a new PartialNW with pre-allocated vectors of the specified size.
    ///
    /// # Parameters
    ///
    /// * `size` - The size to allocate for the dynamic programming vectors.
    pub fn new(size: &usize) -> Self {
        PartialNW { column: vec![0; *size + 1], row: vec![0; *size], best_value: 0 }
    }

    /// Creates a new PartialNW with just the corner element.
    ///
    /// This is typically used to initialize the matrix at the root node.
    pub fn corner() -> PartialNW {
        PartialNW { column: vec![0], row: Vec::new(), best_value: 0 }
    }

    /// Creates a new PartialNW from existing row and column vectors.
    ///
    /// # Parameters
    ///
    /// * `row` - The row vector.
    /// * `col` - The column vector.
    /// * `best` - The best alignment score.
    pub fn from_row_column_best(row: Vec<usize>, col: Vec<usize>, best: &usize) -> PartialNW {
        assert!(row.len() == col.len() - 1);
        PartialNW { column: col, row: row, best_value: *best }
    }

    /// Prints a formatted representation of the partial alignment matrix.
    ///
    /// This method displays the partial dynamic programming matrix for debugging purposes.
    pub fn pretty_print(&self) {
        let row_len = self.row.len();
        let col_len = self.column.len();

        // Sanity check: column should be one longer than row
        assert_eq!(col_len, row_len + 1, "column should be one longer than row");

        // Skip the very first column entry (the "corner") to match the "row" entries
        let width = self.column.len() * 4;

        for elem in &self.column[..self.column.len() - 1] {
            println!("{:>width$}", elem);
        }
        for val in &self.row {
            print!("{:>4}", val); // Right-align with width 4
        }
        print!("{:>4}", &self.column[&self.column.len() - 1]); // Right-align with width 4
        debug!("\nbest value {}\n", self.best_value);
    }
}


impl TrieNode {
    /// Creates a new TrieNode.
    ///
    /// # Parameters
    ///
    /// * `string` - The sequence represented by this node.
    /// * `parent` - Optional reference to the parent node.
    /// * `depth` - The depth of this node in the trie.
    fn new(string: Vec<u8>, parent: Option<Weak<RefCell<TrieNode>>>, depth: &usize) -> Self {
        TrieNode {
            children: HashMap::default(),
            parent: parent,
            visited: 0,
            sequence: string,
            is_terminal: false,
            partial_dp: PartialNW::new(depth),
            depth: *depth,
        }
    }

    /// Updates the alignment data from the parent node.
    ///
    /// This method inherits and extends the dynamic programming matrix from the parent node,
    /// which is efficient for the trie structure where many sequences share prefixes.
    ///
    /// # Parameters
    ///
    /// * `offset_x` - The current position in the sequence.
    /// * `search_sequence` - The sequence being searched for.
    /// * `max_mismatch` - The maximum allowed edit distance.
    pub fn fill_alignment_from_parent(&mut self,
                                     offset_x: &usize,
                                     search_sequence: &[u8],
                                     max_mismatch: &usize) {
        // the order is important - row then column
       let parent_node = match self.parent.as_ref() {
            None => { panic!("Unable to unwrap node at depth {}", self.depth) }
            Some(x) => { x.upgrade() }
        }.unwrap();

        let partial_temp = parent_node.borrow().partial_dp.clone().to_owned();

        self.fill_row_from_partial_nw(&partial_temp, offset_x, search_sequence, max_mismatch);
        self.fill_column_from_partial_nw(&partial_temp, offset_x, search_sequence, max_mismatch);
    }

    /// Fills the alignment matrix for this node.
    ///
    /// # Parameters
    ///
    /// * `row_and_column_basis` - The existing partial alignment matrix to extend.
    /// * `node_offset_x` - The offset into the node's sequence.
    /// * `search_sequence` - The sequence being searched for.
    /// * `max_mismatch` - The maximum allowed edit distance.
    pub fn fill_alignment(&mut self,
                         row_and_column_basis: &PartialNW,
                         node_offset_x: &usize, // This is the node offset into the tree -- i.e. offset 1 == position 0 in the string
                         search_sequence: &[u8],
                         max_mismatch: &usize) {
        
        // the order is important - row then column
        self.fill_row_from_partial_nw(row_and_column_basis, node_offset_x, search_sequence, max_mismatch);
        self.fill_column_from_partial_nw(row_and_column_basis, node_offset_x, search_sequence, max_mismatch);
        
    }

    /// Fills the row of the partial alignment matrix.
    ///
    /// This method calculates the dynamic programming values for the current row
    /// based on the previous partial alignment data.
    ///
    /// # Parameters
    ///
    /// * `row_and_column_basis` - The existing partial alignment matrix to extend.
    /// * `node_offset_x` - The offset into the node's sequence.
    /// * `search_sequence` - The sequence being searched for.
    /// * `max_mismatch` - The maximum allowed edit distance.
    fn fill_row_from_partial_nw(&mut self,
                               row_and_column_basis: &PartialNW,
                               node_offset_x: &usize, // This is the node offset into the tree -- i.e. offset 1 == position 0 in the string
                               search_sequence: &[u8],
                               max_mismatch: &usize) {


        // it's a little confusing -- we can either be in the part of the matrix where each row / column grows by one vs the
        // previous rc_basis (early, before max_mismatch + 1 row/columns) or is the same size as the previous row column (after
        // max_mismatch + 1 rows/columns).
        let mut previous_rc_basis_offset_mm = 0;
        let mut previous_rc_basis_offset_row = 1;
        let mut new_row = vec![*max_mismatch; *max_mismatch];

        // if we're in the early stages do the opposite
        if *max_mismatch + 1 > *node_offset_x {
            previous_rc_basis_offset_mm = 1;
            previous_rc_basis_offset_row = 0;
            new_row = vec![*node_offset_x; *node_offset_x];
        }

        let mut best_value = usize::MAX;

        let search_char = search_sequence[*node_offset_x - 1];
        let comparison_slice = &self.sequence[node_offset_x - new_row.len()..*node_offset_x];
        
        (1..new_row.len()).for_each(|i| {
            let match_mismatched = match search_char == comparison_slice[i - 1] {
                true => { 0 }
                false => { 1 }
            } + row_and_column_basis.row[i - previous_rc_basis_offset_mm];

            let gap_up =
                if i == new_row.len() - 1 { 1 + row_and_column_basis.column[row_and_column_basis.column.len() - 1] } else { 1 + row_and_column_basis.row[i + previous_rc_basis_offset_row] };

            let gap_left = 1 + new_row[i - 1];

            if match_mismatched <= gap_left && match_mismatched <= gap_up {
                new_row[i] = match_mismatched;
                best_value = best_value.min(match_mismatched);
            } else if gap_left < match_mismatched && gap_left < gap_up {
                new_row[i] = gap_left;
                best_value = best_value.min(gap_left);
            } else if gap_up <= match_mismatched && gap_up <= gap_left {
                new_row[i] = gap_up;
                best_value = best_value.min(gap_up);
            } else {
                panic!("Unreachable row state: mm {} gap_up {} gap_left {}", match_mismatched, gap_up, gap_left);
            }
        });
        let new_column = vec![0; new_row.len() + 1];
        self.partial_dp = PartialNW::from_row_column_best(new_row, new_column, &best_value);
    }

    /// Fills the column of the partial alignment matrix.
    ///
    /// This method calculates the dynamic programming values for the current column
    /// based on the previous partial alignment data and the row that was just calculated.
    ///
    /// # Parameters
    ///
    /// * `row_and_column_basis` - The existing partial alignment matrix to extend.
    /// * `node_offset_x` - The offset into the node's sequence.
    /// * `search_sequence` - The sequence being searched for.
    /// * `max_mismatch` - The maximum allowed edit distance.
    fn fill_column_from_partial_nw(&mut self,
                                  row_and_column_basis: &PartialNW,
                                  node_offset_x: &usize,
                                  search_sequence: &[u8],
                                  max_mismatch: &usize) {

        // this function must be called after the row is filled in

        let mut previous_rc_basis_offset = 0;
        let mut new_column = vec![*max_mismatch; *max_mismatch + 1];

        // if we're in the early stages do the opposite
        if *max_mismatch + 1 > *node_offset_x {
            previous_rc_basis_offset = 1;
            new_column = vec![*node_offset_x; *node_offset_x + 1];
        }
        
        let mut best_value = usize::MAX;

        let this_char = self.sequence[*node_offset_x - 1];
        
        let comparison_slice = &search_sequence[(node_offset_x + 1) - new_column.len()..*node_offset_x];

        (1..new_column.len()).for_each(|i| {
            let match_mismatched = match comparison_slice[i - 1] == this_char {
                true => { 0 }
                false => { 1 }
            } + row_and_column_basis.column[i - previous_rc_basis_offset];

            let gap_left = if i == new_column.len() - 1 { 1 + self.partial_dp.row[self.partial_dp.row.len() - 1] } else { 1 + row_and_column_basis.column[(i + 1) - previous_rc_basis_offset] };
            let gap_up = 1 + new_column[i - 1];

        
            if match_mismatched <= gap_left && match_mismatched <= gap_up {
                new_column[i] = match_mismatched;
                best_value = best_value.min(match_mismatched);
            } else if gap_left < match_mismatched && gap_left < gap_up {
                new_column[i] = gap_left;
                best_value = best_value.min(gap_left);
            } else if gap_up < match_mismatched && gap_up <= gap_left {
                new_column[i] = gap_up;
                best_value = best_value.min(gap_up);
            } else {
                panic!("Unreachable col state: mm {} gap_up {} gap_left {}", match_mismatched, gap_up, gap_left);
            }
        });

        self.partial_dp.column = new_column;
        self.partial_dp.best_value = best_value;
    }
}


/// A trie data structure optimized for efficient sequence alignment and searching.
///
/// This implementation includes functionality for efficiently finding sequences
/// within a specified edit distance using partial dynamic programming matrices.
#[derive(Debug)]
pub struct Trie {
    /// Root node of the trie.
    root: Link<TrieNode>,
    
    /// Maximum depth of the trie.
    pub max_height: usize,
    
    /// Maps sequences to their unique identifier.
    sequence_to_id: HashMap<Vec<u8>, usize>,
    
    /// Maps identifiers back to their sequences.
    id_to_sequence: HashMap<usize, Vec<u8>>,
    
    /// Stores nodes by their depth level for efficient level-based traversal.
    depth_links: HashMap<usize, Vec<Link<TrieNode>>>,
    
    /// Counter for assigning sequence identifiers.
    seq_index: usize,
    
    /// Counter for tracking the number of searches performed.
    searches_performed: usize,
    
    /// Current iteration number for search operations.
    iteration: usize,
}

impl Trie {
    /// Creates a new Trie with the specified maximum height.
    ///
    /// # Parameters
    ///
    /// * `max_height` - The maximum depth of the trie.
    pub fn new(max_height: usize) -> Self {
        let sequence_to_id: HashMap<Vec<u8>, usize> = HashMap::default();
        let id_to_sequence: HashMap<usize, Vec<u8>> = HashMap::default();
        let str: Vec<u8> = Vec::new();
        Trie {
            root: Link { 0: Rc::new(RefCell::new(TrieNode::new(str, None, &0))) }, //, //Rc::new(RefCell::new(TrieNode::new(0, 0))),
            max_height,
            sequence_to_id,
            id_to_sequence,
            depth_links: HashMap::default(),
            seq_index: 1,
            searches_performed: 0,
            iteration: 1,
        }
    }

    /// Adds a sequence to the sequence maps and returns its identifier.
    ///
    /// If the sequence already exists, returns its existing identifier.
    ///
    /// # Parameters
    ///
    /// * `seq` - The sequence to add.
    ///
    /// # Returns
    ///
    /// The unique identifier for the sequence.
    pub fn add_sequence(&mut self, seq: &[u8]) -> usize {
        let string_rep = Vec::from(seq);
        let hit = self.sequence_to_id.get(&string_rep);
        match hit {
            None => {
                self.sequence_to_id.insert(string_rep.clone(), self.seq_index);
                self.id_to_sequence.insert(self.seq_index, string_rep);

                self.seq_index += 1;
                self.seq_index - 1
            }
            Some(x) => { *x }
        }
    }

    /// Inserts a sequence into the trie and returns relevant nodes.
    ///
    /// This method builds the trie structure by inserting each character of the sequence,
    /// creating new nodes as needed, and updating the dynamic programming matrices for
    /// sequence alignment calculations.
    ///
    /// # Parameters
    ///
    /// * `seq` - The sequence to insert.
    /// * `depth_to_return` - If provided, returns nodes at this depth encountered during insertion.
    /// * `max_mismatch` - The maximum allowed edit distance for alignment.
    ///
    /// # Returns
    ///
    /// A vector of links to nodes at the requested depth.
    pub fn insert(&mut self, seq: &[u8], depth_to_return: Option<usize>, max_mismatch: &usize) -> Vec<Link<TrieNode>> {
        debug!("Inserting {} with return depth {}",String::from_utf8(seq.to_vec()).unwrap(),depth_to_return.unwrap_or(999));

        let mut current_node = Link { 0: Rc::clone(&self.root) };
        let mut links = Vec::new();

        for i in 0..seq.len() {
            let ch = seq[i];


            if current_node.borrow().children.contains_key(&ch) {
                let pointer_node = Rc::clone(current_node.borrow().children.get(&ch).unwrap());
                if depth_to_return.is_some() && depth_to_return.unwrap() == current_node.borrow().depth {
                    links.push(current_node.clone());
                }
                current_node = Link { 0: Rc::clone(&pointer_node) };
            } else {
                let new_node = Link { 0: Rc::new(RefCell::new(TrieNode::new(seq[0..i + 1].to_vec(), Some(Rc::downgrade(&current_node)), &(i + 1)))) };
                if i == 0 {
                    new_node.0.borrow_mut().fill_alignment(&PartialNW::corner(), &(i + 1), seq, max_mismatch);
                } else {
                    new_node.0.borrow_mut().fill_alignment_from_parent(&(i + 1), seq, max_mismatch);
                }

                if depth_to_return.is_some() && depth_to_return.unwrap() == new_node.borrow().depth { //&& last_real_node.is_some() {
                    links.push(Link { 0: Rc::clone(&new_node) });
                }

                current_node.borrow_mut().children.insert(ch, Rc::clone(&new_node));
                current_node = Link { 0: Rc::clone(&new_node) };

                self.depth_links.entry(i).or_insert_with(Vec::new).push(Link { 0: Rc::clone(&current_node) });
            }

            if i + 1 == self.max_height {
                break;
            }
        }


        // Mark the final node as a terminal
        current_node.borrow_mut().is_terminal = true;
        links
    }

    /// Performs an optimized search through the trie for similar sequences.
    ///
    /// This method implements a chained search strategy that leverages prior work and
    /// partial dynamic programming matrices to efficiently find sequences within
    /// a specified edit distance.
    ///
    /// # Parameters
    ///
    /// * `start_depth` - The depth at which to start the search.
    /// * `future_depth` - If provided, nodes at this depth will be collected for future searches.
    /// * `sequence` - The sequence to search for.
    /// * `max_mismatches` - The maximum allowed edit distance.
    /// * `search_nodes` - The set of nodes to start the search from.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - A vector of matching sequences and their edit distances
    /// - A set of nodes that can be used for future searches
    fn chained_search(&mut self,
                      start_depth: usize,
                      future_depth: Option<usize>,
                      sequence: &[u8],
                      max_mistaches: &usize,
                      search_nodes : &HashSet<Link<TrieNode>>) -> (Vec<(Vec<u8>,usize)>, HashSet<Link<TrieNode>>) {
        let mut hits: Vec<(Vec<u8>,usize)> = Vec::new();
        let mut pebbles: Vec<Link<TrieNode>> = Vec::new(); //HashSet::default();

        let mut current_search_pile: Vec<Link<TrieNode>> = if start_depth < 2 {
            self.depth_links.get(&0).unwrap().iter().map(|x| x.clone()).collect()
        } else {
            search_nodes.iter().map(|x| x.clone()).collect()
        };

        while !current_search_pile.is_empty() {
            let current_node = current_search_pile.pop().unwrap();
            if current_node.borrow().visited < self.iteration { // && current_node.borrow().depth >= start_depth {
                let current_node_depth = current_node.borrow().depth;
        
                current_node.borrow_mut().fill_alignment_from_parent(&(current_node_depth), sequence, max_mistaches);
                current_node.borrow_mut().visited = self.iteration;
        
                self.searches_performed += 1;


                if future_depth.is_some() && current_node_depth < future_depth.unwrap() {
                    pebbles.push(Link { 0: Rc::clone(&current_node) });
                }

                if current_node.borrow().partial_dp.best_value <= *max_mistaches {
                    if current_node.borrow().is_terminal {
                        hits.push((current_node.borrow().sequence.clone(),current_node.borrow().partial_dp.best_value));
                    } else if current_node.borrow().children.len() > 0 {

                        // we're not at the end and children exist, for each child update the DP matrix and add to the pile
                        for child in current_node.borrow().children.values() {
                            current_search_pile.push(Link { 0: Rc::clone(child) });
                        }
                    }
                } else {
                    // we're not going to explore it anymore, but future nodes may need the link
                    pebbles.push(Link { 0: Rc::clone(&current_node.0) });
                }
            }
        }

        // now for each search node, walk back to the future point
        if future_depth.is_some() && future_depth.unwrap() < start_depth {
            let target_depth = if future_depth.unwrap() < *max_mistaches {1} else {future_depth.unwrap() - (max_mistaches)};

            let mut return_pebbles: HashSet<Link<crate::TrieNode>> = HashSet::default();
            pebbles.extend(search_nodes.iter().map(|x| Link { 0: Rc::clone(&x.0) }));

            for nd in pebbles.into_iter() {
                let mut nd_pointer = nd.0.clone();
                while nd_pointer.borrow().depth > target_depth {
                    nd_pointer = Rc::clone(&nd_pointer).borrow_mut().parent.as_ref().unwrap().upgrade().unwrap().clone();
                }
                return_pebbles.insert(Link { 0: nd_pointer });
            }
            self.iteration += 1;
            (hits, return_pebbles)
        } else {
            self.iteration += 1;
            (hits, HashSet::from(pebbles.into_iter().collect()))
        }
    }

    /// Generates a DOT graph representation of the trie.
    ///
    /// This method creates a visualization file in DOT format that can be used
    /// with graphviz tools to visualize the trie structure.
    ///
    /// # Parameters
    ///
    /// * `output_file` - The path to write the DOT file.
    #[allow(dead_code)]
    fn to_dot_plot(&mut self, output_file: &String) {
        let mut file = File::create(output_file).unwrap(); // creates or truncates
        writeln!(file, "graph ER {{").expect("Failed to write dot opening");
        let mut search_nodes = Vec::new();
        search_nodes.push(self.root.clone());

        //println!("explore depth {}",search_nodes.len());
        while !search_nodes.is_empty() {
            let current_node = search_nodes.pop().unwrap();

            writeln!(file, "n{}_d{};", String::from_utf8(current_node.borrow().sequence.clone()).unwrap(), current_node.borrow().depth).expect("Failed to write dot plot entry");

            for child in &current_node.borrow().children {
                debug!("adding pepples");
                let child_node = child.1.borrow();
                writeln!(file, "n{}_d{} -- n{}_d{} [label=\"{}\"];", String::from_utf8(current_node.borrow().sequence.clone()).unwrap(), current_node.borrow().depth,
                         String::from_utf8(child_node.sequence.clone()).unwrap(), child_node.depth,
                         *child.0 as char
                ).expect("Failed to write dot plot entry");
                search_nodes.push(Link { 0: Rc::clone(&child.1) });
            }
        }
        writeln!(file, "}}").expect("Failed to write closing line");
    }

    /// Resets the visited flag for all nodes in the trie.
    ///
    /// This method is used to prepare the trie for a new traversal operation.
    #[allow(dead_code)]
    fn clear_visited(&mut self) {
        let mut search_nodes = Vec::new();
        search_nodes.push(self.root.clone());

        //println!("explore depth {}",search_nodes.len());
        while !search_nodes.is_empty() {
            let current_node = search_nodes.pop().unwrap();
            for child in &current_node.borrow().children {
                search_nodes.push(Link { 0: Rc::clone(&child.1) });
            }
            current_node.borrow_mut().visited = 0;
        }
    }
}


#[derive(Debug)]
struct DistanceGraphNode {
    string: Vec<u8>,
    count: usize,
    valid: bool, // used in the collapsing step
    links: HashMap<Vec<u8>, Weak<RefCell<DistanceGraphNode>>>,
    original_link_count: usize,
    swallowed_links: Vec<(Vec<u8>, usize)>,
}

impl DistanceGraphNode {
    /// Creates a new distance graph node.
    ///
    /// # Parameters
    ///
    /// * `string` - The sequence for this node.
    /// * `count` - The count or frequency of this sequence.
    pub fn new(string: &Vec<u8>, count: &usize) -> DistanceGraphNode {
        DistanceGraphNode { string: string.clone(), count: count.clone(), valid: true, links: HashMap::default(), original_link_count: 0, swallowed_links: Vec::new() }
    }
}


impl Hash for DistanceGraphNode {
    /// Hashes the node based on its sequence.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.string.hash(state);
    }
}

impl PartialEq for DistanceGraphNode {
    /// Compares nodes based on their sequences.
    fn eq(&self, other: &Self) -> bool {
        self.string == other.string
    }
}

impl Eq for DistanceGraphNode {}


/// A graph structure that maintains distances between sequences.
///
/// This structure is used to build and analyze a graph where nodes represent
/// sequences and edges represent edit distances between them.
pub struct LinkedDistances {
    /// Map from sequence to node, with shared ownership.
    nodes: HashMap<Vec<u8>, Link<DistanceGraphNode>>,
}

impl LinkedDistances {
    /// Creates a new, empty LinkedDistances graph.
    #[allow(dead_code)]
    pub fn new() -> LinkedDistances {
        LinkedDistances { nodes: HashMap::default() }
    }

    /// Creates a new LinkedDistances graph from a vector of sequences and their counts.
    ///
    /// # Parameters
    ///
    /// * `strings_and_counts` - Vector of (sequence, count) pairs.
    pub fn new_from_counts(strings_and_counts: &Vec<(Vec<u8>, usize)>) -> LinkedDistances {
        let mut nodes: HashMap<Vec<u8>, Link<DistanceGraphNode>> = HashMap::default();

        for snc in strings_and_counts {
            let dgn = Rc::new(RefCell::new(DistanceGraphNode::new(&snc.0, &snc.1.clone())));
            //println!("Insert {}",String::from_utf8(snc.0.clone().to_vec()).unwrap());
            nodes.insert(snc.0.clone(), Link { 0: dgn });
        }
        LinkedDistances { nodes }
    }

    /// Adds a new node to the graph.
    ///
    /// # Parameters
    ///
    /// * `string` - The sequence for the new node.
    /// * `count` - The count or frequency of this sequence.
    #[allow(dead_code)]
    pub fn add_node(&mut self, string: &Vec<u8>, count: &usize) {
        assert!(!self.nodes.contains_key(string));

        let node = Link { 0: Rc::new(RefCell::new(DistanceGraphNode::new(string, count))) };
        self.nodes.insert(string.clone(), node);
    }

    /// Adds links between nodes in the graph.
    ///
    /// # Parameters
    ///
    /// * `from` - The source sequence.
    /// * `to_nodes` - Vector of (target sequence, edit distance) pairs.
    pub fn add_links(&mut self, from: &Vec<u8>, to_nodes: &Vec<(Vec<u8>,usize)>) {
        //println!("From {}",String::from_utf8(from.to_vec()).unwrap());
        assert!(self.nodes.contains_key(from));


        let from_node = Rc::clone(self.nodes.get(from).as_ref().unwrap());
        for (to_node_name,_count) in to_nodes {
            assert!(self.nodes.contains_key(to_node_name));
            assert_ne!(to_node_name, from);
            let to_node = Rc::clone(self.nodes.get(to_node_name).as_ref().unwrap());

            let entry = self.nodes.entry(from.clone()).
                or_insert(Link { 0: Rc::new(RefCell::new(DistanceGraphNode::new(&Vec::new(), &0))) });

            if !entry.0.borrow().links.contains_key(to_node_name) {
                entry.0.borrow_mut().links.insert(to_node_name.clone(), Rc::downgrade(&to_node));
                let linking = Rc::downgrade(&Rc::clone(&from_node));
                to_node.borrow_mut().links.insert(from.clone(), linking);
            }
            from_node.borrow_mut().original_link_count += 1;
        }
    }

    /// Collapses the graph by merging nodes based on a minimum count ratio.
    ///
    /// This method implements a message passing algorithm that merges lower-count
    /// nodes into higher-count nodes when their ratio exceeds the provided threshold.
    ///
    /// # Parameters
    ///
    /// * `minimum_ratio` - The minimum ratio of counts required for merging.
    ///
    /// # Returns
    ///
    /// A vector of (sequence, node) pairs representing the collapsed graph.
    pub fn message_passing_collpase(self, minimum_ratio: &f64) -> Vec<(Vec<u8>, Link<DistanceGraphNode>)> {

        let mut sorted: Vec<_> = self.nodes.into_iter().collect();

        sorted.sort_by(|a, b| a.1.borrow().count.cmp(&b.1.borrow().count)); // sort by value descending

        let mut modified = true;

        while modified {
            modified = false;

            let mut valid_count = 0;
            sorted.iter().for_each(|x| {
                let valid = x.1.borrow().valid;
                if valid {
                    valid_count += 1;
                    modified = modified | LinkedDistances::message_passing_check(&mut Rc::clone(&x.1), minimum_ratio);
                }
                debug!("Node {} valid {} {}  {}", String::from_utf8(x.0.clone()).unwrap(), valid, x.1.borrow().valid, x.1.borrow().count);
            });
        }
        sorted
    }

    pub fn message_passing_check(link: &mut Rc<RefCell<DistanceGraphNode>>, minimum_ratio: &f64) -> bool {
        let my_count = link.borrow().count;
        let link_size = link.borrow().links.len();

        if link_size > 0 {
            // check that the link doesn't have a self-reference
            let link_name = link.borrow().string.clone();

            let mut link1 = link.borrow_mut();

            link1.links.iter().for_each(|x| if x.0 == &link_name { panic!("Self link {}", String::from_utf8(link_name.clone()).unwrap()) });

            let highest_connection = link1.links.iter().max_by_key(|&(_k, v)| v.upgrade().unwrap().borrow().count).unwrap();

            if highest_connection.1.upgrade().unwrap().borrow().count as f64 / my_count as f64 > *minimum_ratio {

                link1.links.iter().for_each(|dist_link| {
                    let dist = dist_link.1.upgrade().unwrap();
                    dist.borrow_mut().links.remove(&link_name);
                });
                debug!("linker removed! {} {}",
                         String::from_utf8(link1.string.clone()).unwrap(),
                         String::from_utf8(highest_connection.1.clone().upgrade().unwrap().borrow().string.clone()).unwrap());

                // add my count to the larger nodes count
                let sink = highest_connection.1.clone().upgrade().unwrap();
                let mut sink = sink.borrow_mut();
                sink.count += my_count;
                sink.swallowed_links.push((link_name.clone(), my_count));

                debug!("linker removed! {} {} -- {} {}",
                         String::from_utf8(link1.string.clone()).unwrap(),
                         link1.valid,
                         String::from_utf8(highest_connection.1.clone().upgrade().unwrap().borrow().string.clone()).unwrap(),
                         highest_connection.1.clone().upgrade().unwrap().borrow().valid,
                );

                // I'm no longer valid, clear my links too
                link1.valid = false;
                link1.links.clear();

                return true;
            }
        }
        false
    }

    fn prefix_overlap_str(a: &[u8], b: &[u8]) -> usize {
        let ret = a.iter().zip(b.iter())
            .take_while(|(ac, bc)| **ac == **bc)
            .count();
        ret
    }

    fn cluster_string_vector_list(mut strings: Vec<(Vec<u8>, usize)>, max_mismatch: &usize) -> Vec<(Vec<u8>, Link<DistanceGraphNode>)> {
        strings.sort();

        let mut trie = Trie::new(strings.get(0).unwrap().0.len());

        let mut search_nodes = HashSet::default();
        search_nodes.extend(trie.insert(&strings[0].0, Some(1) /* return the first level of the tree */, &max_mismatch));

        // now make a LinkedDistances with the nodes
        let mut linked_dist = LinkedDistances::new_from_counts(&strings);

        (1..strings.len()).for_each(|x| {
            let start = if x > 1 { LinkedDistances::prefix_overlap_str(&strings[x].0, &strings[x - 1].0) } else { 0 };
            let mut future = if x < strings.len() - 1 { LinkedDistances::prefix_overlap_str(&strings[x + 1].0, &strings[x].0) } else { 0 };

            if search_nodes.len() == 0 {
                search_nodes = trie.depth_links.get(&1).unwrap().clone().into_iter().collect();
            }

            if start < strings[0].0.len() {
                let rt = trie.chained_search(start, Some(future), &strings[x].0, &max_mismatch, &search_nodes);
                search_nodes = rt.1;
                linked_dist.add_links(&strings[x].0, &rt.0);

                if future < 1 { future = 1; }

                search_nodes.extend(trie.insert(&strings[x].0, Some(future), &max_mismatch));
            }
        });

        linked_dist.message_passing_collpase(&5.0)
    }
}


#[cfg(test)]
mod tests {
    use std::io;

    use std::io::{BufRead, BufReader};
    use super::*;

    extern crate rand;

    use rand::prelude::*;


    #[allow(dead_code)]
    fn gen_random_dna(len: usize) -> Vec<u8> {
        let nucleotides = vec![b'A', b'C', b'G', b'T'];
        let mut dna = Vec::with_capacity(len);
        for _ in 0..len {
            dna.push(*nucleotides.choose(&mut rand::rng()).unwrap());
        }
        dna
    }

    #[test]
    fn test_insert_one_sequence() {
        let mut trie = Trie::new(10);

        // Insert a single sequence
        trie.insert(&[b'A', b'C', b'G'], None, &2);

        // Navigate the trie manually and check fields
        assert!(trie.root.borrow().children.contains_key(&b'A'));

        let binding = trie.root.borrow();
        let node_a = binding.children.get(&b'A').unwrap();
        assert!(node_a.borrow().children.contains_key(&b'C'));

        let binding = node_a.borrow();
        let node_c = binding.children.get(&b'C').unwrap();
        assert!(node_c.borrow().children.contains_key(&b'G'));

        let binding = node_c.borrow();

        let node_g = binding.children.get(&b'G').unwrap();
        assert!(node_g.borrow().is_terminal);
        // TODO assert_eq!(node_g.borrow().sequence_id, 1); // the null sequence is sequence 0; we're sequence 1
    }

    #[test]
    fn test_insert_multiple_sequences() {
        let mut trie = Trie::new(10);

        // Insert multiple sequences
        trie.insert(&[b'A', b'C', b'G'], None, &2);
        trie.insert(&[b'A', b'C', b'C'], None, &2);
        trie.insert(&[b'T', b'C', b'G', b'A'], None, &2);

        // Check "ACG"
        {
            let binding = trie.root.borrow();
            let node_a = binding.children.get(&b'A').unwrap();
            let binding = node_a.borrow();
            let node_c = binding.children.get(&b'C').unwrap();
            let binding = node_c.borrow();
            let node_g = binding.children.get(&b'G').unwrap();
            assert!(node_g.borrow().is_terminal);
            // TODO assert_eq!(node_g.borrow().sequence_id, 1);
        }

        // Check "ACC"
        {
            let binding = trie.root.borrow();
            let node_a = binding.children.get(&b'A').unwrap();
            let binding = node_a.borrow();
            let node_c = binding.children.get(&b'C').unwrap();
            let binding = node_c.borrow();
            let node_c2 = binding.children.get(&b'C').unwrap();
            assert!(node_c2.borrow().is_terminal);
            // TODO assert_eq!(node_c2.borrow().sequence_id, 2);
        }

        // Check "TGCA"
        {
            let binding = trie.root.borrow();

            let node_t = binding.children.get(&b'T').unwrap();
            let binding = node_t.borrow();

            let node_g = binding.children.get(&b'C').unwrap();
            let binding = node_g.borrow();

            let node_c = binding.children.get(&b'G').unwrap();
            let binding = node_c.borrow();

            let node_a = binding.children.get(&b'A').unwrap();
            assert!(node_a.borrow().is_terminal);
            // TODO assert_eq!(node_a.borrow().sequence_id, 3);
        }
    }


    #[test]
    fn test_overlap_strings() {
        let str1 = vec![b'A', b'A', b'A', b'A', b'A', b'A', b'A', b'A', b'A', b'A', b'G', b'G'];
        let str2 = vec![b'A', b'A', b'A', b'A', b'A', b'A', b'A', b'A', b'A', b'A', b'T', b'T'];
        assert_eq!(LinkedDistances::prefix_overlap_str(str1.as_slice(), str2.as_slice()), 10);
    }

    #[allow(dead_code)]
    fn generate_sequences(length: usize) -> Vec<Vec<u8>> {
        let alphabet = [b'A', b'C', b'G', b'T'];

        if length == 0 {
            return vec![vec![]];
        }

        // Recursively build sequences
        let smaller = generate_sequences(length - 1);
        let mut result = Vec::new();

        for seq in &smaller {
            for ch in alphabet.clone() {
                let mut new_seq = seq.clone();
                new_seq.push(ch);
                result.push(new_seq);
            }
        }

        result
    }

    #[allow(dead_code)]
    fn read_lines_to_vec(path: &str) -> Vec<Vec<u8>> {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);

        let mut buffer: Vec<Vec<u8>> = Vec::new();
        for line in reader.lines() {
            match line {
                Ok(z) => {
                    buffer.push(z.into_bytes());
                }
                Err(_) => { panic!("Problem processing file") }
            }
        }
        // Collect lines into a Vec<String>
        buffer
    }


    fn read_file_to_vec(path: &str) -> io::Result<Vec<(Vec<u8>, usize)>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut results = HashMap::default();

        for line in reader.lines() {
            let line = line?; // Handle any IO error
            if line.trim().is_empty() {
                continue; // Skip blank lines
            }

            // Split on whitespace
            let mut parts = line.split_whitespace();
            let seq_str = parts.next().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing sequence"))?;
            let count_str = parts.next().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing count"))?;

            let seq = seq_str.as_bytes().to_vec(); // Convert to Vec<u8>
            let count: usize = count_str.parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid count"))?;

            results.entry(seq).and_modify(|v| *v += count).or_insert(count);
        }

        Ok(results.into_iter().collect())
    }

    #[test]
    fn test_error_unambiguous_sequences() {
        let mut strings = read_file_to_vec("python/Anchored_error_20mer_set.txt").unwrap();

        let hit_set = LinkedDistances::cluster_string_vector_list(strings,&1);

        // either hits are non error, which should be 120 read counts (100 original reads plus 20 more singletons collapsed into it) or error singletons (1 read)
        for hit in hit_set {
            if hit.1.borrow().count == 120 {
                assert!(hit.1.borrow().valid);
            } else if hit.1.borrow().count == 1 {
                //println!("Hit {} {} {} len {}",String::from_utf8(hit.1.borrow().string.clone()).unwrap(),hit.1.borrow().count,hit.1.borrow().valid,hit.1.borrow().links.len());
                assert!(!hit.1.borrow().valid);
            } else {
                panic!("Unknown result; counts {}",hit.1.borrow().count);
            }
        }
    }


    #[test]
    fn test_first_level() {
        let partial_nw = PartialNW { column: vec![0], row: vec![], best_value: 0 };
        let offset_x = 1;
        let this_string = [b'G', b'T', b'T', b'G', b'C', b'A'];
        let search_sequence = [b'G', b'A', b'T', b'C', b'C', b'A'];
        let max_mismatch = 3;

        let mut trie_node = TrieNode::new(this_string.to_vec(), None, &max_mismatch);

        trie_node.fill_alignment(&partial_nw, &offset_x, &search_sequence, &max_mismatch);
        assert_eq!(trie_node.partial_dp.column, vec![1, 0]);
        assert_eq!(trie_node.partial_dp.row, vec![1]);
    }

    #[test]
    fn test_second_level() {
        let partial_nw = PartialNW { column: vec![1, 0], row: vec![1], best_value: 1 };
        let offset_x = 2;
        let this_string = [b'G', b'T', b'T', b'G', b'C', b'A'];
        let search_sequence = [b'G', b'A', b'T', b'C', b'C', b'A'];
        let max_mismatch = 3;

        let mut trie_node = TrieNode::new(this_string.to_vec(), None, &max_mismatch);

        trie_node.fill_alignment(&partial_nw, &offset_x, &search_sequence, &max_mismatch);
        assert_eq!(trie_node.partial_dp.column, vec![2, 1, 1]);
        assert_eq!(trie_node.partial_dp.row, vec![2, 1]);
    }

    #[test]
    fn test_third_level() {
        let partial_nw = PartialNW { column: vec![2, 1, 1], row: vec![2, 1], best_value: 2 };
        let offset_x = 3;
        let this_string = [b'G', b'T', b'T', b'G', b'C', b'A'];
        let search_sequence = [b'G', b'A', b'T', b'C', b'C', b'A'];
        let max_mismatch = 3;

        let mut trie_node = TrieNode::new(this_string.to_vec(), None, &max_mismatch);

        trie_node.fill_alignment(&partial_nw, &offset_x, &search_sequence, &max_mismatch);
        assert_eq!(trie_node.partial_dp.column, vec![3, 2, 2, 1]);
        assert_eq!(trie_node.partial_dp.row, vec![3, 2, 1]);
    }

    #[test]
    fn test_fourth_level() {
        let partial_nw = PartialNW { column: vec![3, 2, 2, 1], row: vec![3, 2, 1], best_value: 2 };
        let offset_x = 4;
        let this_string = [b'G', b'T', b'T', b'G', b'C', b'A'];
        let search_sequence = [b'G', b'A', b'T', b'C', b'C', b'A'];
        let max_mismatch = 3;

        let mut trie_node = TrieNode::new(this_string.to_vec(), None, &max_mismatch);

        trie_node.fill_alignment(&partial_nw, &offset_x, &search_sequence, &max_mismatch);
        assert_eq!(trie_node.partial_dp.column, vec![3, 3, 2, 2]);
        assert_eq!(trie_node.partial_dp.row, vec![3, 2, 2]);
    }

    #[test]
    fn test_fifth_level() {
        let partial_nw = PartialNW { column: vec![3, 3, 2, 2], row: vec![3, 2, 2], best_value: 2 };
        let offset_x = 5;
        let this_string = [b'G', b'T', b'T', b'G', b'C', b'A'];
        let search_sequence = [b'G', b'A', b'T', b'C', b'C', b'A'];
        let max_mismatch = 3;

        let mut trie_node = TrieNode::new(this_string.to_vec(), None, &max_mismatch);

        trie_node.fill_alignment(&partial_nw, &offset_x, &search_sequence, &max_mismatch);
        assert_eq!(trie_node.partial_dp.column, vec![3, 3, 2, 2]);
        assert_eq!(trie_node.partial_dp.row, vec![3, 3, 3]);
    }

    #[test]
    fn test_sixth_level() {
        let partial_nw = PartialNW { column: vec![3, 3, 2, 2], row: vec![3, 3, 3], best_value: 2 };
        let offset_x = 6;
        let this_string = [b'G', b'T', b'T', b'G', b'C', b'A'];
        let search_sequence = [b'G', b'A', b'T', b'C', b'C', b'A'];
        let max_mismatch = 3;

        let mut trie_node = TrieNode::new(this_string.to_vec(), None, &max_mismatch);

        trie_node.fill_alignment(&partial_nw, &offset_x, &search_sequence, &max_mismatch);
        assert_eq!(trie_node.partial_dp.column, vec![3, 3, 3, 2]);
        assert_eq!(trie_node.partial_dp.row, vec![3, 4, 3]);
    }
}
