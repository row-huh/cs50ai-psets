import os
import random
import re
import sys


DAMPING = 0.85
SAMPLES = 10000



def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")



def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages



def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    from collections import defaultdict

    # Initialize probabilities with default random jump probability
    base_random_prob = (1 - damping_factor) / len(corpus)
    probabilities = defaultdict(lambda: base_random_prob)

    # Handle pages with no outgoing links
    if not corpus[page]:
        return {page_name: 1 / len(corpus) for page_name in corpus}

    # Calculate link-following probability
    link_probability = damping_factor / len(corpus[page])
    
    # Add link-following probabilities to base random probabilities
    for linked_page in corpus[page]:
        probabilities[linked_page] += link_probability

    # Convert defaultdict to regular dict and ensure all corpus pages are included
    return {page_name: probabilities[page_name] for page_name in corpus}


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.
    """
    from collections import defaultdict
    import random

    # Track visit counts for each page
    visit_counts = defaultdict(int)
    pages = list(corpus.keys())
    
    # Initialize with random page
    current_page = random.choice(pages)
    visit_counts[current_page] += 1

    # Perform random walks
    for _ in range(n - 1):
        transitions = transition_model(corpus, current_page, damping_factor)
        
        # Select next page based on transition probabilities
        random_value = random.random()
        cumulative_prob = 0
        
        for page, probability in transitions.items():
            cumulative_prob += probability
            if random_value <= cumulative_prob:
                current_page = page
                break
        
        visit_counts[current_page] += 1

    # Convert visit counts to probabilities
    pageranks = {page: count/n for page, count in visit_counts.items()}
    
    print(f'Sum of sample page ranks: {sum(pageranks.values()):.4f}')
    return pageranks
    
    




def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    """
    from collections import defaultdict
    
    # Initialize constants
    num_pages = len(corpus)
    convergence_threshold = 0.001
    base_probability = (1 - damping_factor) / num_pages
    initial_rank = 1 / num_pages
    iteration_count = 0

    # Initialize pageranks
    current_ranks = {page: initial_rank for page in corpus}
    
    while True:
        iteration_count += 1
        new_ranks = defaultdict(float)

        # Calculate new ranks for each page
        for target_page in corpus:
            rank_sum = 0
            
            # Calculate sum of contributions from pages linking to target_page
            for source_page, links in corpus.items():
                if not links:  # Page with no outlinks
                    rank_sum += current_ranks[source_page] * initial_rank
                elif target_page in links:  # Page with link to target_page
                    rank_sum += current_ranks[source_page] / len(links)
            
            # Apply damping factor formula
            new_ranks[target_page] = base_probability + (damping_factor * rank_sum)

        # Normalize ranks
        rank_sum = sum(new_ranks.values())
        new_ranks = {page: rank/rank_sum for page, rank in new_ranks.items()}

        # Check for convergence
        if all(abs(current_ranks[page] - new_ranks[page]) <= convergence_threshold 
               for page in corpus):
            print(f'Iteration took {iteration_count} iterations to converge')
            print(f'Sum of iteration page ranks: {sum(new_ranks.values()):.4f}')
            return new_ranks

        current_ranks = new_ranks.copy()


if __name__ == "__main__":
    main()