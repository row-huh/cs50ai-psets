import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    
    # for every person
        # list their gene count
        # whether they have a trait 
    
    
    # if they have parents
        # calculate the probability of parent genes
        # name it mother prob and father prob - i think
        # if the parents have 0 gene, then m = 1- m
        # if 1, then m = 0.5
        # else if 2, then m = m where m is the mutation probability
        
        
    # otherwise, given the information you know, fetch the corresponding probabilities,
    # and calculate probabilities for that person
    
    # if im not wrong joint variable that will store the total must go here & probabilities must
    # be added to it at the end of each iteration
    
    
    # setting prob to 1 instead of 0 so that when multiplying by others, the answer does not end up a 0
    joint_probability = 1
    
    
    for person in people:
        person_trait = True if person in have_trait else False
        person_gene = 2 if person in two_genes else 1 if person in one_gene else 0
        person_prob = 1     # initial prob - will change soon as its multiplied

        # if the person has parents then calculate probability otherwise just use the standard ones
        # Mutation doesnt need to be calculated in the mother's and father's case as there is no 'passing down of gene'  
        
        if people[person]['mother'] and people[person]['father']:     
            if people[person]['mother']:
                mother = people[person]['mother']
                mother_trait = True if mother in have_trait else False
                mother_genes = 0
                if mother in two_genes:
                    mother_genes = 2
                elif mother in one_gene:
                    mother_genes = 1
                else:
                    mother_genes = 0
                    
                prob_of_mother_gene = PROBS["gene"][mother_genes]
                prob_of_mother_trait = PROBS["trait"][mother_genes][mother_trait]
                
                mother_prob = prob_of_mother_gene * prob_of_mother_trait
                #joint_probability *= mother_prob


            if people[person]['father']:
                father = people[person]['father']
                father_trait = True if father in have_trait else False
                father_genes = 0
                if father in two_genes:
                    father_genes = 2
                elif father in one_gene:
                    father_genes = 1
                else:
                    father_genes = 0
                    
                prob_of_father_gene = PROBS["gene"][father_genes]
                prob_of_father_trait = PROBS["trait"][father_genes][father_trait]
                
                father_prob = prob_of_father_gene * prob_of_father_trait
                #joint_probability += father_prob
        else:
            prob_of_person_gene = PROBS["gene"][person_gene]
            prob_of_person_trait = PROBS["trait"][person_gene][person_trait]
            
            person_prob *= prob_of_person_trait * prob_of_person_gene
            joint_probability *= person_prob
            continue
        
            
        m = mother_genes
        f = father_genes
        
        mutation = PROBS["mutation"]
        
        if person_gene == 0:
            prob_of_person_gene = (1 - prob_of_father_gene) * (1 - prob_of_mother_gene)
            mutation_prob = (1 - mutation)
        elif person_gene == 1:
            prob_of_person_gene = (prob_of_mother_gene * (1 - prob_of_father_gene)) + ((1 - prob_of_mother_gene) * prob_of_father_gene)
            mutation_prob = 0.5
        else:
            prob_of_person_gene = prob_of_mother_gene * prob_of_father_gene
            mutation_prob = mutation
            
        prob_of_person_trait = PROBS["trait"][person_gene][person_trait]
        person_prob *= prob_of_person_gene * prob_of_person_trait * prob_of_mother_gene * prob_of_father_gene * mutation_prob
        joint_probability *= person_prob
        
    print("Joint probability", joint_probability)
    return joint_probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    raise NotImplementedError


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
def get_probabilities(equation):
    print(equation)