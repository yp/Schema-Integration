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
# Module used to describe a problem solution
#
###

# Standard library
from __future__ import print_function

import logging
import random
import sys


# Personal modules
import util


class Solution:
    
    def __init__(self, instance,
                 clusters):
        assert instance
        self.instance = instance
        self.clusters = clusters
        self.total_similarity = 0.0
        self.cluster_similarities = [ 0.0 ] * len(clusters)
        self.recompute_similarities()

    def is_valid(self):
        valid = True
        assigned = set()
        for cluster in self.clusters:
            for cluster_elem in cluster:
                valid = valid and cluster_elem not in assigned
                assigned.add(cluster_elem)
    
        valid = valid and assigned <= set(range(len(self.instance.names)))
        
        return valid

    def is_complete(self):
        return ( set(( cluster_elem
                       for cluster in self.clusters
                       for cluster_elem in cluster )) ==
                 set(range(len(self.instance.names))) )

    def shuffle_cluster_order(self):
        random.shuffle(self.clusters)
        self.recompute_similarities()

    def sort_clusters_by_similarity(self):
        self.clusters = [ el[1]
                          for el in sorted(enumerate(self.clusters),
                                           key=lambda cl: self.cluster_similarities[cl[0]],
                                reverse=True) ]
        self.cluster_similarities = sorted(self.cluster_similarities,
                                           reverse=True)

    def recompute_similarities(self):
        self.cluster_similarities = [ sum( ( self.instance.similarities[i1][i2]
                                             for i1,i2 in util.lower_triangle(cluster) ) )
                                             for cluster in self.clusters ]
        self.total_similarity = sum(self.cluster_similarities)


    def __str__(self):
        cl_list = [ "([{0:.3f}] {1})".format(cluster_similarity,
                                             ", ".join([str(el) for el in cluster]))
                    for cluster_similarity,cluster
                    in zip(self.cluster_similarities, self.clusters) ]
        return "{0}tot={1:.5f} clusters=[{2}]".format("" if self.is_valid() else "!! INVALID !!",
                                                      self.total_similarity,
                                                      "; ".join(cl_list))


    def logme(self, logfn=logging.debug):
        for cluster_id,(cluster_similarity,
                        cluster) in enumerate(zip(self.cluster_similarities,
                                                  self.clusters)):
            logfn("Cluster #%-3d [%.4f] (%s)", cluster_id, cluster_similarity,
                  ", ".join(( str(el) for el in cluster)) )

    def save_to_file(self, outfile=sys.stdout):
        logging.info("Saving results to file '%s'...", outfile.name)
        outfile.seek(0)
        outfile.truncate(0)
        print('"CLUSTER-ID","SCHEMA-ID","SCHEMA-NAME","#ENTITIES"', file=outfile)
        for cluster_id,cluster in enumerate(self.clusters):
            for schema in cluster:
                print('{cl},{sid},"{sname}",{sent}'.format(cl=cluster_id, sid=schema,
                                                           sname=self.instance.names[schema],
                                                           sent=self.instance.weights[schema]),
                      file=outfile)
        outfile.flush()

