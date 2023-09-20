# A trie is used to map character sequences to bytes and vice versa

from collections import defaultdict

class PrefixTriemap:
    """ Generic, recursive trie structure that is used for finding
    longest prefixes in a pattern.
    """

    def __init__(self, depth=0, max_depth=100):
        """ Creates a new trie (node). 
        
        Parameters:
        -----------
        depth : int
            The depth of the node in the trie.
        """
        self.depth = depth
        self.max_depth = max_depth
        self.children = defaultdict(lambda: PrefixTriemap(depth=self.depth + 1, max_depth=max_depth))
        self.value = None

    def __setitem__(self, pattern, sequence):
        """ Inserts a new pattern into the trie.
        
        Parameters:
        -----------
        pattern : iterable
            The pattern that is to be inserted into the trie.
        sequence : object
            The sequence the pattern maps to.
        """
        if not len(pattern):
            # Leaf node
            self.value = sequence
        else:
            # Recursively insert suffix
            child = self.children[pattern[0]]
            child[pattern[1:]] = sequence
    
    def __getitem__(self, pattern):
        """ Returns the sequence associated with the longest prefix in a pattern.
        
        Parameters:
        -----------
        pattern : iterable
            The pattern the desired sequence is associated with.
        
        Returns:
        --------
        sequence : object
            The desired sequence.
        depth : int
            The size of the longest prefix matched.
        """
        if self.depth > self.max_depth:
            raise Exception(pattern)
        if len(pattern) > 0:
            # Try to match a longer prefix without creating new children
            if pattern[0] in self.children:
                child = self.children[pattern[0]]
                sequence, depth = child[pattern[1:]]
                if sequence is not None:
                    return sequence, depth
        
        # Either the pattern is empty or no longer prefix could be matched.
        return self.value, self.depth
