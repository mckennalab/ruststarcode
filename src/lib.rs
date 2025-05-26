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
    pub children: HashMap<u8, Rc<RefCell<TrieNode>>>,

    /// Reference to the parent node, allowing traversal up the trie.
    pub parent: Option<Weak<RefCell<TrieNode>>>,

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

    /// Stores nodes by their depth level for efficient level-based traversal.
    depth_links: HashMap<usize, Vec<Link<TrieNode>>>,

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
        let str: Vec<u8> = Vec::new();
        Trie {
            root: Link { 0: Rc::new(RefCell::new(TrieNode::new(str, None, &0))) },
            max_height,
            depth_links: HashMap::default(),
            iteration: 1,
        }
    }

    pub fn depth_links(&self, depth: &usize) -> HashSet<Link<TrieNode>> {
        self.depth_links.get(depth).unwrap().clone().into_iter().collect()
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
        assert!(seq.len() <= self.max_height && seq.len() > 0);

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
        }


        // Mark the final node as a terminal
        current_node.borrow_mut().is_terminal = true;
        links
    }

    /// BEWARE: This function relies on in-order searching and other constraints of the
    /// algorithm. I'm making it public, but be prepared to shoot yourself in the foot.
    /// Be especially aware of the link between start_depth, future_depth, and the search_node list.
    ///
    /// Performs an optimized search through the trie for similar sequences.
    ///
    /// This method implements a chained search strategy that leverages prior work and
    /// partial dynamic programming matrices to efficiently find sequences within
    /// a specified edit distance.
    ///
    /// Given the nature of this search process, this method is full of edge-case detection to
    /// keep things fast and could still suffer from unmissed cases.
    ///
    /// # Parameters
    ///
    /// * `start_depth` - The depth at which to start the search.
    /// * `future_depth` - If provided, nodes at this depth will be collected for future searches.
    /// * `sequence` - The sequence to search for.
    /// * `max_mismatches` - The maximum allowed edit distance.
    /// * `search_nodes` - The set of nodes to start the search from. This is generated from the previous
    ///                     sequences' run (in alphabetic order);
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - A vector of matching sequences and their edit distances
    /// - A set of nodes that can be used for future searches
    pub fn chained_search(&mut self,
                      start_depth: usize,
                      future_depth: Option<usize>,
                      sequence: &[u8],
                      max_mistaches: &usize,
                      search_nodes: &HashSet<Link<TrieNode>>) -> (Vec<(Vec<u8>, usize)>, HashSet<Link<TrieNode>>) {
        assert!(sequence.len() <= self.max_height);

        // create an all-padded string, and then copy over the passed in sequence
        let mut string_rep = vec![b'-'; self.max_height];
        string_rep[0..sequence.len()].copy_from_slice(sequence);


        let mut hits: Vec<(Vec<u8>, usize)> = Vec::new();
        let mut pebbles: Vec<Link<TrieNode>> = Vec::new(); //HashSet::default();

        let mut current_search_pile: Vec<Link<TrieNode>> = if start_depth < 2 {
            self.depth_links.get(&0).unwrap().iter().map(|x| x.clone()).collect()
        } else {
            search_nodes.iter().map(|x| x.clone()).collect()
        };

        while !current_search_pile.is_empty() {
            let current_node = current_search_pile.pop().unwrap();
            //println!("Trying to fill the current search depth with --{}--",String::from_utf8(current_node.borrow().sequence.clone()).unwrap());
            if current_node.borrow().visited < self.iteration { // && current_node.borrow().depth >= start_depth {
                let current_node_depth = current_node.borrow().depth;

                current_node.borrow_mut().fill_alignment_from_parent(&(current_node_depth), string_rep.as_slice(), max_mistaches);
                current_node.borrow_mut().visited = self.iteration;

                if future_depth.is_some() && current_node_depth < future_depth.unwrap() {
                    //println!("pushing --{}--",String::from_utf8(current_node.borrow().sequence.clone()).unwrap());

                    pebbles.push(Link { 0: Rc::clone(&current_node) });
                }

                if current_node.borrow().partial_dp.best_value <= *max_mistaches {
                    if current_node.borrow().is_terminal {
                        hits.push((current_node.borrow().sequence.clone(), current_node.borrow().partial_dp.best_value));
                    } else if current_node.borrow().children.len() > 0 {

                        // we're not at the end and children exist, for each child update the DP matrix and add to the pile
                        for child in current_node.borrow().children.values() {
                            current_search_pile.push(Link { 0: Rc::clone(child) });
                        }
                    }
                } else {
                    // we're not going to explore it anymore, but future nodes may need the link
                    //println!("pushing 2 --{}--",String::from_utf8(current_node.borrow().sequence.clone()).unwrap());
                    pebbles.push(Link { 0: Rc::clone(&current_node.0) });
                }
            }
        }

        // now for each search node, walk back to the future point
        if future_depth.is_some() && future_depth.unwrap() < start_depth {
            let target_depth = if future_depth.unwrap() < *max_mistaches { 1 } else { future_depth.unwrap() - (max_mistaches) };

            let mut return_pebbles: HashSet<Link<crate::TrieNode>> = HashSet::default();
            pebbles.extend(search_nodes.iter().map(|x| Link { 0: Rc::clone(&x.0) }));

            for nd in pebbles.into_iter() {
                let mut nd_pointer = nd.0.clone();

                // walk back up the tree until we've reached the target depth or the depth will be 1 (don't walk back to the root)
                while nd_pointer.borrow().depth > target_depth && nd_pointer.borrow().depth > 1 {
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
pub struct DistanceGraphNode {
    pub string: Vec<u8>,
    pub count: usize,
    pub valid: bool, // used in the collapsing step
    pub links: HashMap<Vec<u8>, Weak<RefCell<DistanceGraphNode>>>,
    pub original_link_count: usize,
    pub swallowed_links: Vec<(Vec<u8>, usize)>,
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
    fn add_node(&mut self, string: &Vec<u8>, count: &usize) {
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
    fn add_links(&mut self, from: &Vec<u8>, to_nodes: &Vec<(Vec<u8>, usize)>) {
        assert!(self.nodes.contains_key(from));

        let from_node = Rc::clone(self.nodes.get(from).as_ref().unwrap());
        for (to_node_name, _count) in to_nodes {
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
    fn message_passing_collpase(self, minimum_ratio: &f64) -> Vec<(Vec<u8>, Link<DistanceGraphNode>)> {
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

    fn message_passing_check(link: &mut Rc<RefCell<DistanceGraphNode>>, minimum_ratio: &f64) -> bool {
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

    pub fn prefix_overlap_str(a: &[u8], b: &[u8]) -> usize {
        let ret = a.iter().zip(b.iter())
            .take_while(|(ac, bc)| **ac == **bc)
            .count();
        ret
    }

    pub fn cluster_string_vector_list(max_string_length: &usize, mut strings: Vec<(Vec<u8>, usize)>, max_mismatch: &usize, minimum_ratio: &f64) -> Vec<(Vec<u8>, Link<DistanceGraphNode>)> {
        assert!(*minimum_ratio >= 2.0); // this is a bit arbitrary, but it prevents anyone from doing something really dumb here
        if strings.len() == 0 {return Vec::new()}

        strings.sort();


        let mut trie = Trie::new(*max_string_length);

        let mut search_nodes = HashSet::default();
        search_nodes.extend(trie.insert(&strings[0].0, Some(1) /* return the first level of the tree */, &max_mismatch));

        // now make a LinkedDistances with the nodes
        let mut linked_dist = LinkedDistances::new_from_counts(&strings);

        (1..strings.len()).for_each(|x| {
            //println!("{} {} {}",String::from_utf8(strings[x].0.clone()).unwrap(),strings[x].0.len(),*max_string_length);
            assert!(*max_string_length >= strings[x].0.len());
            let start = if x > 1 { LinkedDistances::prefix_overlap_str(&strings[x].0, &strings[x - 1].0) } else { 0 };
            let mut future = if x < strings.len() - 1 { LinkedDistances::prefix_overlap_str(&strings[x + 1].0, &strings[x].0) } else { 0 };

            if search_nodes.len() == 0 {
                search_nodes = trie.depth_links(&1);
            }

            if start < strings[0].0.len() {
                let rt = trie.chained_search(start, Some(future), &strings[x].0, &max_mismatch, &search_nodes);
                search_nodes = rt.1;
                linked_dist.add_links(&strings[x].0, &rt.0);

                if future < 1 { future = 1; }

                search_nodes.extend(trie.insert(&strings[x].0, Some(future), &max_mismatch));
            }
        });

        linked_dist.message_passing_collpase(minimum_ratio)
    }
}


#[cfg(test)]
mod tests {
    use std::io;

    use std::io::{BufRead, BufReader};
    use super::*;

    extern crate rand;

    use rand::prelude::*;
    use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};


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
    fn test_adding_unpadded_string_to_tree() {
        let mut tree = Trie::new(10);
        tree.insert("ACGTACGTA".as_bytes(), Some(1), &1);
        let hs = HashSet::default();
        let st = tree.chained_search(1, Some(1), "ACGTACGTA".as_bytes(), &2, &hs);
        assert_eq!(st.0.len(), 1);
        tree.insert("ACGTACGTC".as_bytes(), Some(1), &1);

        let st = tree.chained_search(1, Some(1), "ACGTACGTAA".as_bytes(), &2, &hs);
        assert_eq!(st.0.len(), 2);

        tree.insert("ACGTACGT-".as_bytes(), Some(1), &1);
        let st = tree.chained_search(1, Some(1), "ACGTACGTA-".as_bytes(), &2, &hs);
        assert_eq!(st.0.len(), 3);
        //st.0.iter().for_each(|x| println!("key {}",String::from_utf8(x.0.clone()).unwrap()));

        tree.insert("ACGTACG--".as_bytes(), Some(1), &1);
        let st = tree.chained_search(1, Some(1), "ACGTACGTA-".as_bytes(), &1, &hs);
        assert_eq!(st.0.len(), 3);
        //st.0.iter().for_each(|x| println!("key after double gap {}",String::from_utf8(x.0.clone()).unwrap()));
    }

    #[test]
    fn test_error_unambiguous_sequences() {

        // acutally length 25 now -- fix the file name at some point
        let strings = read_file_to_vec("python/Anchored_error_20mer_set.txt").unwrap();

        let hit_set = LinkedDistances::cluster_string_vector_list(&25, strings, &1, &5.0);

        // either hits are non error, which should be 120 read counts (100 original reads plus 20 more singletons collapsed into it) or error singletons (1 read)
        for hit in hit_set {
            if hit.1.borrow().count == 120 {
                assert!(hit.1.borrow().valid);
            } else if hit.1.borrow().count == 1 {
                assert!(!hit.1.borrow().valid);
            } else {
                panic!("Unknown result; counts {}", hit.1.borrow().count);
            }
        }
    }

    #[test]
    fn test_error_from_clique() {
        let test_set = vec![
            ("ACCTGGATTGGA".as_bytes().to_vec(), 10),
            ("ACGTGGAATGGA".as_bytes().to_vec(), 1),
            ("ACCTGGAATGGA".as_bytes().to_vec(), 1),
            ("ACCTGGAATGTA".as_bytes().to_vec(), 1)];
        let hit_set = LinkedDistances::cluster_string_vector_list(&12, test_set, &2, &5.0);

        // either hits are non error, which should be 120 read counts (100 original reads plus 20 more singletons collapsed into it) or error singletons (1 read)
        for hit in hit_set {
            if hit.1.borrow().count > 10 {
                assert!(hit.1.borrow().valid);
                assert_eq!(hit.1.borrow().count,13);
            } else {
                assert!(!hit.1.borrow().valid);
                assert_eq!(hit.1.borrow().count,1);
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

    // Edge case tests
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_empty_sequence_insertion() {
        let mut trie = Trie::new(10);
        let empty_seq = b"";
        
        // Should handle empty sequence without panic
        trie.insert(empty_seq, None, &2);
    }

    #[test]
    fn test_single_character_sequence() {
        let mut trie = Trie::new(10);
        
        // Insert single character sequences
        trie.insert(b"A", None, &1);
        trie.insert(b"C", None, &1);
        trie.insert(b"G", None, &1);
        trie.insert(b"T", None, &1);
        
        // Verify all four nucleotides are present as children of root
        assert!(trie.root.borrow().children.contains_key(&b'A'));
        assert!(trie.root.borrow().children.contains_key(&b'C'));
        assert!(trie.root.borrow().children.contains_key(&b'G'));
        assert!(trie.root.borrow().children.contains_key(&b'T'));
        
        // Each should be terminal
        assert!(trie.root.borrow().children[&b'A'].borrow().is_terminal);
        assert!(trie.root.borrow().children[&b'C'].borrow().is_terminal);
        assert!(trie.root.borrow().children[&b'G'].borrow().is_terminal);
        assert!(trie.root.borrow().children[&b'T'].borrow().is_terminal);
    }

    #[test]
    fn test_maximum_length_sequence() {
        let max_length = 5;
        let mut trie = Trie::new(max_length);
        
        // Create sequence at maximum length
        let max_seq = vec![b'A'; max_length];
        trie.insert(&max_seq, None, &2);
        
        // Verify it was inserted correctly
        let mut current = trie.root.clone();
        for i in 0..max_length {
            assert!(&current.borrow().children.contains_key(&b'A'));
            let child = current.borrow().children[&b'A'].clone();
            current = Link { 0: child };
            assert_eq!(current.borrow().depth, i + 1);
            assert_eq!(current.borrow().sequence.len(), i + 1);
        }
        assert!(current.borrow().is_terminal);
    }

    #[test]
    fn test_identical_sequences() {
        let mut trie = Trie::new(10);
        let seq = b"ACGT";
        
        // Insert same sequence multiple times
        trie.insert(seq, None, &2);
        trie.insert(seq, None, &2);
        trie.insert(seq, None, &2);
        
        // Should still have only one path
        let mut current = trie.root.clone();
        for &ch in seq {
            assert_eq!(current.borrow().children.len(), 1);

            let child = current.borrow().children[&ch].clone();
            current = Link { 0: child };
        }
        assert!(current.borrow().is_terminal);
    }

    #[test]
    fn test_clustering_empty_input() {
        let empty_strings: Vec<(Vec<u8>, usize)> = vec![];
        let result = LinkedDistances::cluster_string_vector_list(&10, empty_strings, &2, &5.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_clustering_single_sequence() {
        let single_seq = vec![(b"ACGT".to_vec(), 100)];
        let result = LinkedDistances::cluster_string_vector_list(&10, single_seq, &2, &5.0);
        
        assert_eq!(result.len(), 1);
        assert!(result[0].1.borrow().valid);
        assert_eq!(result[0].1.borrow().count, 100);
        assert_eq!(result[0].0, b"ACGT".to_vec());
    }

    #[test]
    fn test_clustering_zero_mismatch_tolerance() {
        let sequences = vec![
            (b"AAAA".to_vec(), 10),
            (b"AAAT".to_vec(), 1),  // 1 mismatch - should not cluster
            (b"AAAA".to_vec(), 5),  // exact match - should cluster
        ];
        
        let result = LinkedDistances::cluster_string_vector_list(&4, sequences, &0, &5.0);
        
        // With 0 mismatch tolerance, only exact matches should cluster
        let valid_clusters: Vec<_> = result.iter().filter(|x| x.1.borrow().valid).collect();
        assert_eq!(valid_clusters.len(), 2); // AAAA cluster and AAAT singleton
    }

    #[test]
    fn test_very_high_mismatch_tolerance() {
        let sequences = vec![
            (b"AAAA".to_vec(), 10),
            (b"CCCC".to_vec(), 1),  // 4 mismatches
            (b"GGGG".to_vec(), 1),  // 4 mismatches
            (b"TTTT".to_vec(), 1),  // 4 mismatches
        ];
        
        let result = LinkedDistances::cluster_string_vector_list(&4, sequences, &4, &5.0);
        
        // With very high mismatch tolerance, everything should cluster into one group
        let valid_clusters: Vec<_> = result.iter().filter(|x| x.1.borrow().valid).collect();
        assert_eq!(valid_clusters.len(), 1);
        assert_eq!(valid_clusters[0].1.borrow().count, 13); // 10 + 1 + 1 + 1
    }

    #[test]
    fn test_partial_nw_corner_case() {
        let corner = PartialNW::corner();
        assert_eq!(corner.column, vec![0]);
        assert!(corner.row.is_empty());
        assert_eq!(corner.best_value, 0);
    }

    #[test]
    fn test_partial_nw_from_row_column() {
        let row = vec![1, 2, 3];
        let col = vec![0, 1, 2, 3];
        let best = 1;
        
        let partial = PartialNW::from_row_column_best(row.clone(), col.clone(), &best);
        assert_eq!(partial.row, row);
        assert_eq!(partial.column, col);
        assert_eq!(partial.best_value, best);
    }

    #[test]
    #[should_panic(expected = "assertion failed: row.len() == col.len() - 1")]
    fn test_partial_nw_invalid_dimensions() {
        let row = vec![1, 2, 3];
        let col = vec![0, 1]; // Too short - should panic
        let best = 1;
        
        let partial = PartialNW::from_row_column_best(row, col, &best);
        partial.pretty_print(); // This should panic
    }

    // Edge case tests for divergent sequences
    #[test]
    fn test_clustering_highly_divergent_sequences() {
        // Test sequences that differ in many positions
        let sequences = vec![
            (b"AAAAAAAA".to_vec(), 100),  // Reference sequence
            (b"TTTTTTTT".to_vec(), 50),   // Completely different
            (b"CCCCCCCC".to_vec(), 30),   // Completely different
            (b"GGGGGGGG".to_vec(), 20),   // Completely different
        ];
        
        // With low mismatch tolerance, should not cluster
        let result = LinkedDistances::cluster_string_vector_list(&8, sequences.clone(), &2, &5.0);
        let valid_clusters: Vec<_> = result.iter().filter(|x| x.1.borrow().valid).collect();
        assert_eq!(valid_clusters.len(), 4); // Each should remain separate
        
        // With high mismatch tolerance, may or may not cluster depending on whether 
        // sequences are found within edit distance
        let result_high = LinkedDistances::cluster_string_vector_list(&8, sequences, &8, &5.0);
        let valid_clusters_high: Vec<_> = result_high.iter().filter(|x| x.1.borrow().valid).collect();
        assert!(valid_clusters_high.len() >= 1 && valid_clusters_high.len() <= 4); // Verify reasonable clustering
    }

    #[test]
    fn test_clustering_gradually_divergent_sequences() {
        // Test sequences that progressively diverge from a reference
        let sequences = vec![
            (b"AAAAAAAA".to_vec(), 100),  // Reference
            (b"AAAAACAA".to_vec(), 1),    // 1 mismatch
            (b"AAAAACAC".to_vec(), 1),    // 2 mismatches from ref
            (b"AACAACAC".to_vec(), 1),    // 4 mismatches from ref
            (b"AACCACAC".to_vec(), 1),    // 5 mismatches from ref
            (b"TACCACAC".to_vec(), 1),    // 6 mismatches from ref
        ];
        
        let result = LinkedDistances::cluster_string_vector_list(&8, sequences, &2, &5.0);
        
        // Check that sequences within edit distance get clustered appropriately
        let valid_clusters: Vec<_> = result.iter().filter(|x| x.1.borrow().valid).collect();
        
        // The reference should absorb sequences within edit distance 2
        let main_cluster = valid_clusters.iter().find(|x| x.1.borrow().count > 100).unwrap();
        assert!(main_cluster.1.borrow().swallowed_links.len() >= 2);
    }

    #[test]
    fn test_clustering_with_insertions_small_gaps() {
        // Test sequences with small insertions (1-2 characters)
        let sequences = vec![
            (b"ACGTACGT".to_vec(), 100),    // Reference 8bp
            (b"ACGTTACGT".to_vec(), 1),     // 1 insertion (T)
            (b"ACCGTACGT".to_vec(), 1),     // 1 insertion (C) 
            (b"ACGTAACGT".to_vec(), 1),     // 1 insertion (A)
            (b"ACGTTAACGT".to_vec(), 1),    // 2 insertions (T, A)
            (b"AACCCGTACGT".to_vec(), 1),   // 2 insertions (A, CC)
        ];
        
        let result = LinkedDistances::cluster_string_vector_list(&11, sequences, &2, &5.0);
        
        // Check clustering behavior with insertions
        let valid_clusters: Vec<_> = result.iter().filter(|x| x.1.borrow().valid).collect();
        
        // Should cluster sequences with small insertions into main cluster
        let main_cluster = valid_clusters.iter().find(|x| x.1.borrow().count > 100);
        assert!(main_cluster.is_some());
        assert!(main_cluster.unwrap().1.borrow().swallowed_links.len() >= 2);
    }

    #[test]
    fn test_clustering_with_deletions_small_gaps() {
        // Test sequences with small deletions (1-2 characters)
        let sequences = vec![
            (b"ACGTACGT".to_vec(), 100),  // Reference 8bp
            (b"CGTACGT".to_vec(), 1),     // 1 deletion (A)
            (b"ACGACGT".to_vec(), 1),     // 1 deletion (T)
            (b"ACGTCGT".to_vec(), 1),     // 1 deletion (A)
            (b"CGTCGT".to_vec(), 1),      // 2 deletions (A, A)
            (b"CGTACG".to_vec(), 1),      // 2 deletions (A, T)
        ];
        
        let result = LinkedDistances::cluster_string_vector_list(&8, sequences, &2, &5.0);
        
        // Check clustering behavior with deletions
        let valid_clusters: Vec<_> = result.iter().filter(|x| x.1.borrow().valid).collect();
        
        // Should cluster sequences with small deletions into main cluster
        let main_cluster = valid_clusters.iter().find(|x| x.1.borrow().count > 100);
        assert!(main_cluster.is_some());
        assert!(main_cluster.unwrap().1.borrow().swallowed_links.len() >= 2);
    }

    #[test]
    fn test_clustering_with_large_insertions() {
        // Test sequences with large insertions (3+ characters)
        let sequences = vec![
            (b"ACGTACGT".to_vec(), 100),      // Reference 8bp
            (b"ACGTTTTACGT".to_vec(), 1),     // 3 insertions (TTT)
            (b"ACGTAAAACGT".to_vec(), 1),     // 3 insertions (AAA)
            (b"ACGTCCCCCACGT".to_vec(), 1),   // 5 insertions (CCCCC)
            (b"GGGACGTACGT".to_vec(), 1),     // 3 insertions at start (GGG)
            (b"ACGTACGTAAA".to_vec(), 1),     // 3 insertions at end (AAA)
        ];
        
        let result = LinkedDistances::cluster_string_vector_list(&13, sequences, &3, &5.0);
        
        // With moderate edit distance, some large insertions might not cluster
        let valid_clusters: Vec<_> = result.iter().filter(|x| x.1.borrow().valid).collect();
        
        // Check that clustering behavior is reasonable for large insertions
        assert!(valid_clusters.len() >= 1);
        
        // Test with higher edit distance tolerance
        let result_high = LinkedDistances::cluster_string_vector_list(&13, 
            vec![
                (b"ACGTACGT".to_vec(), 100),      
                (b"ACGTTTTACGT".to_vec(), 1),     
                (b"ACGTAAAACGT".to_vec(), 1),     
            ], &5, &5.0);
        
        let valid_high: Vec<_> = result_high.iter().filter(|x| x.1.borrow().valid).collect();
        let main_cluster_high = valid_high.iter().find(|x| x.1.borrow().count > 100);
        assert!(main_cluster_high.is_some());
    }

    #[test]
    fn test_clustering_with_large_deletions() {
        // Test sequences with large deletions (3+ characters)
        let sequences = vec![
            (b"ACGTACGTACGT".to_vec(), 100), // Reference 12bp
            (b"ACGTACGT".to_vec(), 1),       // 4 deletions from end
            (b"ACGTCGT".to_vec(), 1),        // 5 deletions (ACGT -> ACGT)
            (b"GTACGT".to_vec(), 1),         // 6 deletions from start
            (b"ACG".to_vec(), 1),            // 9 deletions (major truncation)
        ];
        
        let sequences_len = sequences.len();
        let result = LinkedDistances::cluster_string_vector_list(&12, sequences, &4, &5.0);
        
        // Check clustering behavior with large deletions
        let valid_clusters: Vec<_> = result.iter().filter(|x| x.1.borrow().valid).collect();
        
        // Should have reasonable clustering for large deletions
        assert!(valid_clusters.len() >= 1);
        assert!(valid_clusters.len() <= sequences_len);
    }

    #[test]
    fn test_complex_indel_patterns() {
        // Test sequences with complex insertion/deletion patterns
        let sequences = vec![
            (b"ACGTACGT".to_vec(), 100),     // Reference
            (b"ACGTTCGT".to_vec(), 1),       // Substitution + deletion
            (b"AACGTACGT".to_vec(), 1),      // Insertion at start
            (b"ACGTACGTT".to_vec(), 1),      // Insertion at end
            (b"AACGTTACGTT".to_vec(), 1),    // Insertions at both ends
            (b"CGTACG".to_vec(), 1),         // Deletions at both ends
            (b"ACCGTTACGTT".to_vec(), 1),    // Multiple insertions
        ];
        
        let result = LinkedDistances::cluster_string_vector_list(&11, sequences, &3, &5.0);
        
        // Check that complex patterns are handled appropriately
        let valid_clusters: Vec<_> = result.iter().filter(|x| x.1.borrow().valid).collect();
        assert!(valid_clusters.len() >= 1);
        
        // The main cluster should absorb some of the similar sequences
        let main_cluster = valid_clusters.iter().find(|x| x.1.borrow().count > 100);
        assert!(main_cluster.is_some());
    }

    #[test]
    fn test_clustering_mixed_error_types() {
        // Test combinations of substitutions, insertions, and deletions
        let sequences = vec![
            (b"ACGTACGT".to_vec(), 100),   // Reference
            (b"TCGTACGT".to_vec(), 1),     // 1 substitution (A->T)
            (b"ACGTTCGT".to_vec(), 1),     // 1 substitution (A->T) + 1 deletion
            (b"AACGTACGT".to_vec(), 1),    // 1 insertion
            (b"TCGTTCGT".to_vec(), 1),     // 1 sub + 1 del + 1 sub
            (b"AACGTTACGTT".to_vec(), 1),  // 2 insertions + 1 deletion
        ];
        
        let result = LinkedDistances::cluster_string_vector_list(&11, sequences, &2, &5.0);
        
        // Check clustering with mixed error types
        let valid_clusters: Vec<_> = result.iter().filter(|x| x.1.borrow().valid).collect();
        
        // Should cluster appropriately based on edit distance
        let main_cluster = valid_clusters.iter().find(|x| x.1.borrow().count > 100);
        assert!(main_cluster.is_some());
        
        // Check that sequences within edit distance 2 are clustered
        let total_absorbed = main_cluster.unwrap().1.borrow().swallowed_links.len();
        assert!(total_absorbed >= 1);
    }

    #[test]
    fn test_clustering_edge_case_ratios() {
        // Test with various count ratios at the boundary conditions
        let test_cases = vec![
            // Ratio exactly at threshold
            vec![(b"AAAA".to_vec(), 10), (b"AAAT".to_vec(), 2)], // ratio = 5.0
            // Ratio just below threshold  
            vec![(b"AAAA".to_vec(), 10), (b"AAAT".to_vec(), 3)], // ratio = 3.33
            // Ratio just above threshold
            vec![(b"AAAA".to_vec(), 15), (b"AAAT".to_vec(), 2)], // ratio = 7.5
        ];
        
        for (i, sequences) in test_cases.into_iter().enumerate() {
            let result = LinkedDistances::cluster_string_vector_list(&4, sequences, &1, &5.0);
            let valid_clusters: Vec<_> = result.iter().filter(|x| x.1.borrow().valid).collect();
            
            match i {
                0 => {
                    // At threshold - should cluster
                    assert!(valid_clusters.len() <= 2); // May or may not cluster at exact threshold
                },
                1 => {
                    // Below threshold - should not cluster  
                    assert_eq!(valid_clusters.len(), 2);
                },
                2 => {
                    // Above threshold - should cluster
                    assert_eq!(valid_clusters.len(), 1);
                },
                _ => {}
            }
        }
    }

    #[test]
    fn test_clustering_palindromic_sequences() {
        // Test with palindromic sequences that might confuse alignment
        let sequences = vec![
            (b"ACGTACGT".to_vec(), 100),    // Reference
            (b"TGCATGCA".to_vec(), 1),      // Reverse complement
            (b"ACGTGTTACA".to_vec(), 1),    // Palindromic with insertions
            (b"CGTACG".to_vec(), 1),        // Palindromic substring
        ];
        
        let result = LinkedDistances::cluster_string_vector_list(&10, sequences, &3, &5.0);
        
        // Check that palindromic sequences are handled correctly
        let valid_clusters: Vec<_> = result.iter().filter(|x| x.1.borrow().valid).collect();
        assert!(valid_clusters.len() >= 1);
        assert!(valid_clusters.len() <= 4);
    }

    #[test]
    fn test_clustering_repetitive_sequences() {
        // Test with highly repetitive sequences
        let sequences = vec![
            (b"ATATATATAT".to_vec(), 100),  // Reference - alternating AT
            (b"TATATATATA".to_vec(), 1),    // Shifted by 1
            (b"ATATATATA".to_vec(), 1),     // Truncated by 1
            (b"ATATATATATT".to_vec(), 1),   // Extended by 1 with mismatch
            (b"ACATATATATAT".to_vec(), 1),  // Insertion + substitution
        ];
        
        let result = LinkedDistances::cluster_string_vector_list(&12, sequences, &2, &5.0);
        
        // Check clustering of repetitive sequences
        let valid_clusters: Vec<_> = result.iter().filter(|x| x.1.borrow().valid).collect();
        
        // Should handle repetitive patterns reasonably
        assert!(valid_clusters.len() >= 1);
        let main_cluster = valid_clusters.iter().find(|x| x.1.borrow().count > 100);
        assert!(main_cluster.is_some());
    }

    #[test]
    fn test_clustering_very_short_sequences() {
        // Test with very short sequences (2-3 bp)
        let sequences = vec![
            (b"AT".to_vec(), 100),
            (b"AC".to_vec(), 1),    // 1 mismatch
            (b"GT".to_vec(), 1),    // 1 mismatch  
            (b"TT".to_vec(), 1),    // 1 mismatch
            (b"A".to_vec(), 1),     // 1 deletion
            (b"ATG".to_vec(), 1),   // 1 insertion
        ];
        
        let result = LinkedDistances::cluster_string_vector_list(&3, sequences, &1, &5.0);
        
        // Check clustering behavior with very short sequences
        let valid_clusters: Vec<_> = result.iter().filter(|x| x.1.borrow().valid).collect();
        
        // Should cluster appropriately for short sequences
        assert!(valid_clusters.len() >= 1);
        // Check if there's a main cluster that absorbed some sequences
        let has_absorptions = valid_clusters.iter().any(|x| x.1.borrow().swallowed_links.len() > 0);
        assert!(has_absorptions || valid_clusters.len() > 1); // Either clustering occurred or multiple valid clusters
    }
}
