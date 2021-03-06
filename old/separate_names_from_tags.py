from __future__ import division, print_function
from collections import defaultdict
from glob import glob
from operator import and_

freqdict = defaultdict(int)

road_tags = ["Road", "Avenue", "Street", "Drive", "Lane",
             "Crescent", "Walk", "Park", "Terrace", "Close", "Link",
             "Place", "Way", "Grove", "Rise", "View", "Hill", "Estate",
             "Farmway", "Green", "Garden", "Gardens", "Junction", "Boulevard",
             "Central", "Circle", "Court", "Loop", "Track", "Square",
             "Heights", "Village", "Promenade", "Vale", "Cross", "Vista",
             "Sector", "Circus", "Bridge", "Gate", "Valley", "Turn",
             "Interchange", "Plaza", "Little", "Mount", "Highway", "Quay",
             "Mall", "Bank", "Plain", "Beach", "Height", "Wood", "Ring",
             "Ridge", "Island", "Ind", "Industrial", "Terminal", "Coast",
             "Centre", "Northview", "Reservoir", "Alley "]

malay_prefix_tags = ["Jalan", "Lorong", "Bukit", "Lengkok", "Taman"]


def mean_word_length(name):
    assert type(name) == list
    lengths = [len(word) for word in name]
    return sum(lengths) / len(lengths)

# open dictionary
dictionary = dict()
for dictfile in glob('resources/scowl-7.1/final/english-words*'):
    if dictfile.endswith('95'):
        continue
    with open(dictfile, 'r') as g:
        for line in g.readlines():
            dictionary[line.strip()] = 1

with open('sg_roadnames.txt', 'r') as f:
    for line in f:
        if line.startswith('##'):
            continue
        words = line.strip().split()

        # remove integers/A-Z
        words = [word for word in words if not word[0].isdigit()
                 and not (len(word) > 1 and word[1].isdigit())
                 and not word.startswith(':')
                 and not len(word) == 1]

        # remove "North/South/East/West/Central/Upper"
        words = [word for word in words if not word in
                 ["North", "South", "East", "West", "Central", "Upper", "Old"]]

        # now split road tag from the actual name
        tags = []
        malay_tag = 0
        # occasionally there may be >1 tag e.g. "Ring Road"
        while len(words) >= 2 and words[-1] in road_tags:
            tags.append(words[-1])  # wrong order, we'll reverse them later
            remainder = words[:-1]
            words.pop()
        if not tags:
            if words[0] in malay_prefix_tags:
                tags.append(words[0])
                remainder = words[1:]
                malay_tag = 1  # we'll use this as a feature
            else:
                tags = [""]
                remainder = words

        # another feature: are all the words in the name in the dictionary?
        # good indicator that it's a 'generic' name, like 'Market St'
        all_words_in_dict = 1 if reduce(and_, [word.lower() in dictionary
                                               for word in remainder]) else 0

        # print result and some basic features
        print(line.strip(),                     # full road name
              ' '.join(reversed(tags)),         # road tags
              ' '.join(remainder),              # the actual name of the road
              malay_tag,                        # whether the road tag is Malay
              mean_word_length(remainder),      # mean word length of the name
              1 if all_words_in_dict else 0,    # are all words in dict?
              sep="\t")
