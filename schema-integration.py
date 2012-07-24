#!/usr/bin/env python

##########
#
#                          SCHEMA-INTEGRATION
#  A clustering approach to guide DB schema integration processes
#
#  Copyright (C) 2012  Yuri Pirola, Riccardo Dondi
#
#  Distributed under the terms of the GNU General Public License (GPL)
#
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##########


from __future__ import print_function

import argparse
import collections
import copy
import itertools
import logging
import math
import random
import sys

from schemaintegration.util import positive_integer

import schemaintegration.schemas
import schemaintegration.ilpformulation
import schemaintegration.solution
import schemaintegration.solver
import schemaintegration.cplexsolver
import schemaintegration.gurobisolver

# Ensure replicability
random.seed(26343)


def compute_initial_partition(schemas, constraints):
    n = len(schemas.names)
    # Try to compute a good starting point
    #  - phase 1: check if schemas.weights are too high
    assert ( sum(sorted(schemas.weights, reverse=True)[:constraints.ll])
             <= constraints.eu )
    assert n >= 2*constraints.lu

    #  - phase 2: split the input set
    partition = [ list(range(starting_point,
                             min(starting_point+constraints.lu, n)))
                  for starting_point in range(0, n, constraints.lu)]
    i = 0
    while len(partition[-1]) < constraints.ll:
        partition[-1].append(partition[i][-1])
        partition[i] = partition[i][:-1]
        i = (i + 1) % (len(partition)-1)

    if len(partition) % 2 == 1:
        new_set = []

        while len(partition[i]) <= constraints.ll:
            i = (i + 1) % len(partition)

        while len(new_set) < constraints.ll:
            new_set.append(partition[i][-1])
            partition[i] = partition[i][:-1]
            i = (i + 1) % len(partition)
            while len(partition[i]) <= constraints.ll:
                i = (i + 1) % len(partition)

        partition.append(new_set)

    return schemaintegration.solution.Solution(schemas, partition)



class InstanceGenerator:

    def __init__(self, solution, max_instance_size):
        self._solution = solution
        self._max_size = max_instance_size
        self._can_reuse = False

    def generator(self):
        self._partial_set = None
        for cluster in self._solution.clusters:
            if not self._partial_set:
                self._partial_set = list(cluster)
                continue
            if ( len(self._partial_set) + len(cluster) > self._max_size ):
                ret_elems = list(self._partial_set)
                self._partial_set = []
                self._can_reuse = True
                yield ret_elems
                assert ( len(self._partial_set) + len(cluster) <=
                         self._max_size )
            self._partial_set.extend(cluster)
        self._can_reuse = False
        while self._partial_set:
            ret_elems = list(self._partial_set)
            self._partial_set = []
            yield ret_elems

    def can_reuse(self):
        return self._can_reuse

    def reuse(self, elems):
        assert self._can_reuse
        logging.debug("Reusing %s", elems)
        self._partial_set.extend(elems)



parser = argparse.ArgumentParser()
parser.add_argument('input-file',
                    nargs='?',
                    type=argparse.FileType(mode='r'),
                    default=sys.stdin)
parser.add_argument('output-file',
                    nargs='?',
                    type=argparse.FileType(mode='w'),
                    default=sys.stdout)
parser.add_argument('-l', '--min-cardinality',
                    help="the minimum cardinality of each cluster",
                    type=positive_integer, required=True)
parser.add_argument('-u', '--max-cardinality',
                    help="the maximum cardinality of each cluster",
                    type=positive_integer, required=True)
parser.add_argument('-e', '--max-entities',
                    help="the maximum number of entities in each cluster",
                    type=positive_integer, required=True)
parser.add_argument('-s', '--similarity-matrix',
                    help="the file where the computed schema similarity "
                    "matrix will be written to",
                    type=argparse.FileType(mode='w'))
parser.add_argument('-g', '--similarity-graph',
                    help="the file where the computed schema similarity "
                    "graph will be written to",
                    type=argparse.FileType(mode='w'))
parser.add_argument('-v', '--verbose',
                    help='increase output verbosity',
                    action='count', default=0)

args = parser.parse_args()
args = vars(args)

if args['verbose'] == 0:
    log_level = logging.INFO
elif args['verbose'] == 1:
    log_level = logging.DEBUG
else:
    log_level = logging.DEBUG


logging.basicConfig(level=log_level,
                    format='%(levelname)-8s [%(asctime)s]  %(message)s',
                    datefmt="%y%m%d %H%M%S")


logging.info("Schema integration via clustering")
logging.info("Copyright (C) 2012 Yuri Pirola, Riccardo Dondi")
logging.info("This program is distributed under the terms of the GNU General Public License (GPL), v3 or later.")
logging.info("This program comes with ABSOLUTELY NO WARRANTY. See the GNU General Public License for more details.")
logging.info("This is free software, and you are welcome to redistribute it under the conditions specified by the license.")

infile = args["input-file"]
outfile = args["output-file"]


schemas = schemaintegration.schemas.read_and_prepare_instance(infile)

schemas.print_stats(logging.debug)

if ( 'similarity_matrix' in args and
     args['similarity_matrix'] is not None ):
    schemas.save_to_csv(args['similarity_matrix'])

if ( 'similarity_graph' in args and
     args['similarity_graph'] is not None ):
    schemas.transform_similarities_to_graphviz(args['similarity_graph'])


constr = schemaintegration.ilpformulation.Constraints(
    ll= args['min_cardinality'],
    lu= args['max_cardinality'],
    eu= args['max_entities'])

logging.debug("%s", constr)


solution = compute_initial_partition(schemas, constr)

logging.debug("Initial element partition:")
solution.logme(logging.debug)

# Solve indipendently for each pair of the partition
best_solution = None

max_instance_size = constr.lu + 2*constr.ll

max_cardinality_increase = constr.lu + constr.ll
tries_for_cardinality = int(3*max_cardinality_increase/2)


current_try = 0
current_increase = 0


formulation = schemaintegration.ilpformulation.StrictILPFormulation(
    schemas, constr )

while True:
    new_solution = schemaintegration.solution.Solution(schemas, [])

    solution.shuffle_cluster_order()
    gen = InstanceGenerator(solution, max_instance_size)
    for elems in gen.generator():

        solver = schemaintegration.solver.Solver(
            schemaintegration.gurobisolver.GurobiSolver() )

        formulation.prepare_ILP(solver, elems, k="auto")

        solver.solve()

        (sol_status, partial_solution) = formulation.get_solution()

        if sol_status in [ schemaintegration.solver.SolutionStatus.INFEASIBLE,
                           schemaintegration.solver.SolutionStatus.INVALID ]:
            logging.warn("Something went wrong!! Skipping this iteration.")
            break

        good_clusters = len(partial_solution.clusters)
        if gen.can_reuse():
            # Reorder clusters
            ## Commented because is not really useful, since the process stucks to some
            ## local best clusters.
            # partial_solution.sort_clusters_by_similarity()

            # Get the latest clusters until max_instance_size-lu elements
            last_elems = 0
            while ( good_clusters > 0 and
                    ( last_elems +
                      len(partial_solution.clusters[good_clusters-1])
                      < max_instance_size - constr.lu ) ):
                good_clusters = good_clusters - 1
                last_elems = last_elems + len(partial_solution.clusters[good_clusters])
            # Re-clusterize the others
            gen.reuse([ el
                        for cluster in partial_solution.clusters[good_clusters:]
                        for el in cluster ])
        # The heaviest clusters are the first ones, save it
        new_solution.clusters.extend(partial_solution.clusters[:good_clusters])

        new_solution.recompute_similarities()
        logging.info("Partial similarity: %.4f", new_solution.total_similarity)


    new_solution.recompute_similarities()
    logging.info("Total similarity: %.4f", new_solution.total_similarity)

    if ( new_solution and
         ( not best_solution or
           ( best_solution and
             best_solution.total_similarity < new_solution.total_similarity ) ) ):
        logging.info("New clusterization improves the old one "
                     "(new obj= %.5f, old obj= %.5f).",
                     new_solution.total_similarity,
                     best_solution.total_similarity if best_solution else 0.0)
        best_solution = new_solution
        logging.info("Best clusterization so far:")
        best_solution.logme(logging.info)
        best_solution.save_to_file(outfile)
        solution = copy.copy(best_solution)
    else:
        logging.info("New clusterization does not improve the old one "
                     "(new obj= %.5f, old obj= %.5f).",
                     new_solution.total_similarity if new_solution else 0.0,
                     best_solution.total_similarity if best_solution else 0.0)

        if current_try < tries_for_cardinality:
            current_try = current_try + 1
            logging.info("Re-trying with the same instance size limit. "
                         "Remaining tries: %d",
                         tries_for_cardinality - current_try)
        elif current_increase < max_cardinality_increase:
            current_try = 0
            current_increase = current_increase + 1
            max_instance_size = max_instance_size + 1
            tries_for_cardinality = max(1, tries_for_cardinality - 1)
            logging.info("Increasing instance size limit to %d. "
                         "Remaining increases: %d",
                         max_instance_size,
                         max_cardinality_increase - current_increase)
        else:
            logging.info("No remaining tries. Ending...")
            break


best_solution.save_to_file(outfile)


logging.info("Schema integration via clustering -- Completed")




