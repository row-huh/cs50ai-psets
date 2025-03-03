import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())


    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """

        
        for variable in self.domains:
            length = variable.length

            words = self.domains[variable]
            new = self.domains[variable].copy()

            for word in words:
                if len(word) != length:
                    new.remove(word)

            self.domains[variable] = new


    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        overlap = self.crossword.overlaps[x, y]
        

        if overlap is None:
            return False
            
        x_idx, y_idx = overlap  # posistins where overlap occurs
        
        to_remove = set()
        
        # For each word in x's domain
        for word_x in self.domains[x]:

            if not any( word_x[x_idx] == word_y[y_idx] for word_y in self.domains[y]):
                to_remove.add(word_x)
                revised = True
                
        self.domains[x] -= to_remove
        
        return revised



    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            queue = [] # using list as a queue
            for x in self.domains:
                for y in self.crossword.neighbors(x):
                    queue.append((x, y))
        else:
            queue = list(arcs)
        
        while queue:
            x, y = queue.pop(0)  
            
            if self.revise(x, y):
                
                if len(self.domains[x]) == 0:
                    return False
                
                for z in self.crossword.neighbors(x):
                    if z != y:
                        queue.append((z, x))
        
        return True


        
    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for variable in self.crossword.variables:
            if variable not in assignment:
                return False
    
    
        return True



    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # Check for duplicate words
        words_used = set()
        for var, word in assignment.items():
            # If this word has already been used, assignment is inconsistent
            if word in words_used:
                return False
            words_used.add(word)
            
            # Check word length matches variable length
            if var.length != len(word):
                return False
            
            # Check for conflicts at overlapping positions
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    # Get the overlap between the variables
                    overlap = self.crossword.overlaps[var, neighbor]
                    if overlap:
                        var_idx, neighbor_idx = overlap
                        # Check if the letters at the overlap match
                        if word[var_idx] != assignment[neighbor][neighbor_idx]:
                            return False
        
        # If no inconsistencies found, the assignment is consistent
        return True


    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        ruled_out_counts = {}
        
        for word in self.domains[var]:
            count = 0
            
            if var in assignment:
                continue
                
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    continue
                    
                overlap = self.crossword.overlaps[var, neighbor]
                if overlap is None:
                    continue
                    
                var_idx, neighbor_idx = overlap
                
                for neighbor_word in self.domains[neighbor]:
                    if word[var_idx] != neighbor_word[neighbor_idx]:
                        count += 1
                        
            ruled_out_counts[word] = count
    
        return sorted(self.domains[var], key=lambda word: ruled_out_counts[word])


    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # Find all unassigned variables
        unassigned = [v for v in self.crossword.variables if v not in assignment]
        
        if not unassigned:
            return None
        
        # Choose the variable with the minimum remaining values (MRV)
        # If tied, use degree heuristic (most constraints with other variables)
        def key_function(var):
            # Primary sort: minimum remaining values
            # Secondary sort: degree (number of neighbors)
            return (len(self.domains[var]), -len(self.crossword.neighbors(var)))
        
        return min(unassigned, key=key_function)


    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment
            
        var = self.select_unassigned_variable(assignment)
        
        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value
            
            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result
        
        return None



def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
