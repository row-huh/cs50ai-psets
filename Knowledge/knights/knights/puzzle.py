from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
Astatement = And(AKnave, AKnight)
knowledge0 = And(
    # A is either a knave or a knight but not both
    And(Or(AKnave, AKnight), Not(And(AKnave, AKnight))), 
    # if a is a knave, what he said is a lie
    Implication(AKnave, Not(Astatement)),
    # if a is a knight, what he said is the truth
    Implication(AKnight, Astatement)
)


# Puzzle 1
# A says "We are both knaves."
# B says nothing.
Astatement = And(AKnave, BKnave)
knowledge1 = And(
    # A is either a knave or a knight but not both
    And(Or(AKnave, AKnight), Not(And(AKnave, AKnight))),
    # B is either a knave or a knight but not both
    And(Or(BKnave, BKnight), Not(And(BKnave, BKnight))),
    # if a is a knight, what he said is the truth
    Implication(AKnight, Astatement),
    # if a is a knave, what he said is not true
    Implication(AKnave, Not(Astatement)),
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
Astatement = And(Implication(AKnave, BKnave), Implication(AKnight, BKnight))
Bstatement = And(Implication(BKnave, AKnight), Implication(BKnight, AKnave))
knowledge2 = And(
    # A is either a knave or a knight but not both
    And(Or(AKnave, AKnight), Not(And(AKnave, AKnight))),
    # B is either a knave or a knight but not both
    And(Or(BKnave, BKnight), Not(And(BKnave, BKnight))),
    # if a is a knight, what he said is the truth
    Implication(AKnight, Astatement),
    # if a is a knave, what he said is not true
    Implication(AKnave, Not(Astatement)),
    # if b is a knight, what he said is the truth
    Implication(BKnight, Bstatement),
    # if b is a knave, what he said is not true
    Implication(BKnave, Not(Bstatement)),
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
Astatement = Or(AKnave, AKnight)
Bstatement = AKnave
Bstatement2 = CKnave
Cstatement = AKnight
knowledge3 = And(
    # A is either a knave or a knight but not both
    And(Or(AKnave, AKnight), Not(And(AKnave, AKnight))),
    # B is either a knave or a knight but not both
    And(Or(BKnave, BKnight), Not(And(BKnave, BKnight))),
    # C is either a knave or a knight but not both
    And(Or(CKnave, CKnight), Not(And(CKnave, CKnight))),
    # if a is a knight, what he said is the truth
    Implication(AKnight, Astatement),
    # if a is a knave, what he said is not true
    Implication(AKnave, Not(Astatement)),
    # if b is a knight, what he said is the truth
    Implication(BKnight, And(Bstatement, Bstatement2)),
    # if b is a knave, what he said is not true
    Implication(BKnave, Not(And(Bstatement, Bstatement2))),
    # if c is a knight, what he said is the truth
    Implication(CKnight, Cstatement),
    # if C is a knave, what he said is not true
    Implication(CKnave, Not(Cstatement)),
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()