from __future__ import division, print_function
from collections import defaultdict

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
             "Centre", "Northview", "Reservoir", "Alley"]

malay_prefix_tags = ["Jalan", "Lorong", "Bukit", "Lengkok", "Taman"]

with open('../sg_roadnames.txt', 'r') as f:
    for line in f:
        if line.startswith('##'):
            continue
        words = line.strip().split()

        # -------------------
        #  Remove qualifiers
        # -------------------

        # remove integers/A-Z
        words = [word for word in words if not word[0].isdigit()
                 and not (len(word) > 1 and word[1].isdigit())
                 and not word.startswith(':')
                 and not len(word) == 1]

        # remove "North/South/East/West/Central/Upper"
        words = [word for word in words if not word in
                 ["North", "South", "East", "West", "Central", "Upper", "Old"]]

        # -------------------
        #  Identify road tag
        # -------------------

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

        # print result and some basic features
        print(line.strip(),                     # full road name
              ' '.join(remainder),              # the actual name of the road
              ' '.join(reversed(tags)),         # road tags
              sep="\t")