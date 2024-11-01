import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")      

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]    

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells (which are mines),
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        # if the number of known cells is equal to number of mines
        if len(self.cells) == self.count:
            return self.cells
        else:
            return set()
        

            
    def known_safes(self):
        #TODO
        """
        Returns the set of all cells in self.cells known to be safe.
        """

        if self.count == 0:
            return self.cells
        else:
            return set()
            
    
    def mark_mine(self, cell):
        #TODO
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.count -= 1
            self.cells.remove(cell)
        else:
            return

    def mark_safe(self, cell):
        #TODO
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)
        else:
            return


class MinesweeperAI():
    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []
    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)
    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)


    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        

        # 1) Mark the cell as a move that has been made
        self.moves_made.add(cell)
        
        # 2) Mark the cell as safe, updating any sentences that contain the cell as well
        if cell not in self.safes:
            self.mark_safe(cell)
        for sentence in self.knowledge:
            for cell in sentence.cells:
                if cell in self.safes:
                    sentence.cells.remove(cell)
            
        # 3) Add a new sentence to the AI's knowledge base based 
        # on the value of `cell` and `count` 
        # if count of cell is 0 - then all the neighbors of that cell are safe
        # if the count of cell is not 0 - then neighbors of that cell = count #
        
        neighbors = self.get_neighbors(cell)
        if count == 0:
            for neighbor in neighbors:
                self.mark_safe(neighbor)
        elif len(neighbors) == count:
            for neighbor in neighbors:
                self.mark_mine(neighbor) 
        else:
            # if any neighbor is a known mine then modify the sentence from
            # {A, B, C} = 2 - out of which lets say A is a mine - to
            # {B, c} = 1
            neighbors -= self.safes
            for neighbor in neighbors:
                if neighbor in self.mines:
                    count -= 1
            neighbors -= self.mines
            
            self.knowledge.append(Sentence(neighbors, count))
          

      
        
            
        # my program works and actually solves the game, i'm guessing it's my logic, 
        # we know there are mines when the number of cells are equal to the count, 
        # we know all cells are safe when the count is zero, we know 
        # set2 - set1  = count2 - count1 if set1 is a subsest of set2 
        # and i'm looping these checks whenever there's a change to the knowledge base #
        

        


    def get_neighbors(self, cell):
        neighbors = set()
        
        i, j = cell
        
        rows = range(i-1, i+2)
        cols = range(j-1, j+2)
        
        height = range(self.height)
        width = range(self.width)
        
        for row in rows:
            if row in height:
                for col in cols:
                    print(row, col)
                    if col in width:
                        if (row, col) not in self.moves_made:
                            neighbors.add((row, col))
        neighbors -= self.safes
        neighbors -= self.moves_made
        return neighbors
    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for move in self.safes:
            if move not in self.moves_made:
                self.moves_made.add(move)
                return move
    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """

        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.moves_made and (i, j) not in self.mines:
                    return (i, j)