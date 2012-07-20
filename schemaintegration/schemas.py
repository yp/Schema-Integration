##########
#
#                          SCHEMA-INTEGRATION
#  A clustering approach to guide DB schema integration processes
#
#  Copyright (C) 2012  Yuri Pirola, Riccardo Dondi
#
#  Distributed under the terms of the GNU General Public License (GPL)
#
#  This file is part of SCHEMA-INTEGRATION.
#
#  SCHEMA-INTEGRATION is free software: you can redistribute it and/or
#  modify it under the terms of the GNU General Public License as
#  published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#
#  SCHEMA-INTEGRATION is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with SCHEMA-INTEGRATION.
#  If not, see <http://www.gnu.org/licenses/>.
#
##########

###
#
# Module used to describe an input instance
#
###

# Standard library
from __future__ import print_function

import logging
import sys


# Personal modules
import util


class Schemas:

    def __init__(self):
        self.names = []
        self.entities = []
        self.entity_similarities = {}
        self.weights = None
        self.similarities = None
        self._prepared = False



    def _prepare_instance(self):
        def compute_similarity(entity_similarities):
            return float( sum([ sum(row) for row in entity_similarities ]) /
                          (len(entity_similarities)+len(entity_similarities[0])) )

        assert self is not None
        assert self.names
        assert len(self.names) == len(self.entities)

        if self._prepared:
            return

        # Polish schema names
        ext = '.graphml'
        self.names = [ name[:-len(ext)] if name.endswith(ext) else name
                       for name in self.names ]

        # Prepare weights: weights[schema_i] = no. of entities
        logging.info("Computing schema weights...")
        self.weights = [ len(entity_set) for entity_set in self.entities ]

        # Compute weights
        logging.info("Computing schema similarities...")
        n_schemas = len(self.names)
        # - prepare matrix
        self.similarities = [ [ 0.0 ] * n_schemas for i in xrange(n_schemas) ]

        # - compute raw similarities
        max_similarity = 0.0
        for i1,i2 in util.lower_triangle(n_schemas):
            this_similarity = compute_similarity(self.entity_similarities[(i1,i2)])
            self.similarities[i1][i2] = this_similarity
            self.similarities[i2][i1] = this_similarity
            max_similarity = max(max_similarity, this_similarity)

        logging.debug("Maximum similarity between two schemas: %.4f", max_similarity)
        assert max_similarity > 0.0
        #  - rescale matrix
        self.similarities = [ [ el/max_similarity for el in row ]
                              for row in self.similarities ]

        self._prepared = True



    def print_stats(self, logfn=logging.info):
        self._prepare_instance()
        assert self._prepared

        logfn("Schemas (%d):", len(self.names))
        for sn,sw in zip(self.names, self.weights):
            logfn("  - schema [entities=%3d]: '%s'", sw, sn)
        logfn("Most similar schemas:")
        n_most_similar = 2
        for sn,ss in zip(self.names, self.similarities):
            logfn("  - schema: '%s'", sn)
            for msn, mss in sorted(zip(self.names, ss),
                                   key=lambda x: x[1],
                                   reverse=True)[:n_most_similar]:
                logfn("      * [similarity=%.4f] '%s'", mss, msn)



    def save_to_csv(self, outfile=sys.stdout):
        self._prepare_instance()
        assert self._prepared

        logging.info("Saving the computed similarity matrix to file '%s'...",
                     outfile.name)
        # First row:      <space>,schema names
        # Second row:     <space>,schema weights (ie, no. of entities)
        # Remaining rows: <schema name>,similarities
        print('"","{0}"'.format('","'.join(self.names)),
              file=outfile)
        print('"",{0}'.format(','.join([str(w) for w in self.weights])),
              file=outfile)
        for sn,simrow in zip(self.names, self.similarities):
            print('"{0}",{1}'.format(sn, ','.join([ '{0:.6f}'.format(simel)
                                                    for simel in simrow ])),
                  file=outfile)
        outfile.flush()



    def transform_similarities_to_graphviz(self,
                                           outfile=sys.stdout,
                                           min_visible_similarity=0.15,
                                           min_label_similarity=0.30):
        self._prepare_instance()
        assert self._prepared
        assert 0 <= min_visible_similarity <= min_label_similarity

        logging.info("Represent similarities as a graph to file '%s'...",
                     outfile.name)

        print('graph similarities {', file=outfile)
        # Preparing nodes
        print('  node [shape="rect", fontsize=14];', file=outfile)
        for i,(sn,sw) in enumerate(zip(self.names, self.weights)):
            print('  s{0} [label="{1}\\n[{2}]"];'.format(i, sn, sw),
                  file=outfile)
        # Preparing edges
        print('  edge [len=5, fontsize=10];', file=outfile)
        for i1,i2,simil in [ (i1, i2, self.similarities[i1][i2])
                             for i1,i2 in util.lower_triangle(len(self.names))
                             if self.similarities[i1][i2] >= min_visible_similarity ]:
            edge_format = ( 'label="{0:.3f}",style=solid'.format(simil) if simil > min_label_similarity
                            else 'style=dashed,constraint=false' )
            print('  s{0} -- s{1} [{2}];'.format(i1, i2, edge_format),
                  file=outfile)
        print('}', file=outfile)
        outfile.flush()



def read_and_prepare_instance(infile):
    assert infile
    schemas = _read_from_file(infile)
    assert schemas
    schemas._prepare_instance()
    return schemas




def _read_from_file(infile):
    logging.info("Reading schemas from '%s'...", infile.name)

    schemas = Schemas()

    next_is_header = True
    next_is_first = False

    for line in infile:
        line = line.strip().split(",")
        if not any(line): # If the line is empty, set to go to the next schema
            next_is_header = True
            header = None
            continue

        # Strip empty cells
        line = [ cell for cell in line if cell ]

        if next_is_header:
            # The first cell must be "SCHEMAi VS SCHEMAj"
            header = line[0].strip('"')
            # logging.debug("Read header '%s'...", header)
            assert " VS " in header
            header = header.split(" VS ")
            assert len(header)==2

            if header[0] not in schemas.names:
                id0 = len(schemas.names)
                schemas.names.append(header[0])
                schemas.entities.append([])
                add_schema0 = True
            else:
                id0 = schemas.names.index(header[0])
                add_schema0 = False

            if header[1] not in schemas.names:
                id1 = len(schemas.names)
                schemas.names.append(header[1])
                schemas.entities.append([])
                add_schema1 = True
            else:
                id1 = schemas.names.index(header[1])
                add_schema1 = False

            ids = tuple(sorted([id0, id1]))
            assert ids not in schemas.entity_similarities
            schemas.entity_similarities[ids] = []

            next_is_header = False
            next_is_first = True

        elif next_is_first:
            # Read the entity names
            if add_schema1:
                schemas.entities[id1] = [ name.strip('"') for name in line ]
                add_schema1 = False

            assert schemas.entities[id1] == [ name.strip('"') for name in line ]
            next_is_first = False
            next_row = 0

        else:
            # Read a new row of the matrix
            if add_schema0:
                schemas.entities[id0].append(line[0].strip('"'))
            assert schemas.entities[id0][next_row] == line[0].strip('"')
            current_row = [ float(x) for x in line[1:] if x ]
            assert len(current_row) == len(schemas.entities[id1])
            schemas.entity_similarities[ids].append(current_row)
            next_row = next_row + 1

    logging.info("Read %d schemas.", len(schemas.names))

    schemas._prepared = False

    return schemas



