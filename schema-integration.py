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
import itertools
import logging
import math
import sys

import cplex

class LogFile:
    def __init__(self, prefix=""):
        self._buff= []
        self._prefix= prefix

    def _do_print(self):
        logging.debug("%s%s",
                      self._prefix,
                      "".join(self._buff))
        self._buff= []

    def __enter__(self):
        if self._buff:
            self._do_print()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._buff:
            self._do_print()
        if exc_type is not None:
            logging.warn("%sERROR: %s", self._prefix, exc_value)
        return False

    def write(self, string):
        outstr= "{0}".format(string)
        if "\n" not in outstr:
            self._buff.append(outstr)
        else:
            lines= outstr.split("\n", 1)
            assert len(lines)==2
            self._buff.append(lines[0])
            self._do_print()
            self.write(lines[1])
        return self

    def flush(self):
        return self

def positive_integer(string):
    try:
        value = int(string)
    except:
        raise argparse.ArgumentTypeError("'{0}' is not a positive integer.".format(string))
    if value <= 0:
        raise argparse.ArgumentTypeError("'{0}' is not a positive integer.".format(string))
    return value

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

logging.info("Reading schemas from '%s'...", infile.name)

schema_names = set()
schema_entities = {}
next_is_header = True
next_is_first = False
weights = {}
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

        schema_names.update(header)

        if header[0] not in weights:
            weights[header[0]] = {}
        assert header[1] not in weights[header[0]]

        current_weights = []
        weights[header[0]][header[1]] = current_weights

        if header[0] not in schema_entities:
            header0_entities = []
            schema_entities[header[0]] = header0_entities
        else:
            header0_entities = None

        next_is_header = False
        next_is_first = True

    elif next_is_first:
        # Read the entity names
        if header[1] not in schema_entities:
            schema_entities[header[1]] = [ name.strip('"') for name in line ]
        assert schema_entities[header[1]] == [ name.strip('"') for name in line ]
        next_is_first = False

    else:
        # Read a new row of the matrix
        if header0_entities is not None:
            header0_entities.append(line[0].strip('"'))
        current_weights.append([ float(x) for x in line[1:] if x ])

logging.info("Read %d schemas.", len(schema_names))
maxsnlen = max(( len(sn) for sn in schema_names))
for sn in schema_names:
    logging.info("  - schema: '%s',%s  %3d entities",
                 sn, ' ' * (maxsnlen-len(sn)), len(schema_entities[sn]))

logging.info("Computing schema similarities...")

schema_names = list(schema_names)
schema_entities = [ schema_entities[sn] for sn in schema_names ]

for n1 in schema_names:
    if n1 not in weights:
        weights[n1]= {}
    for n2 in schema_names:
        if n1==n2:
            continue
        if n2 not in weights:
            weights[n2]= {}
        if n2 in weights[n1]:
            weights[n2][n1] = weights[n1][n2]
        else:
            assert n1 in weights[n2]
            weights[n1][n2] = weights[n2][n1]


no_of_schemas = len(schema_names)

matrix = [ [0]*no_of_schemas for i in xrange(no_of_schemas) ]

def compute_similarity(n1, n2, e1, e2, w):
    return sum([ sum(row) for row in w ])/(len(e1)+len(e2))
#    return sum([ sum(row) for row in w ])/(len(e1)*len(e2))

max_similarity = 0.0
for r,n1 in enumerate(schema_names):
    for c,n2 in enumerate(schema_names):
        if c >= r:
            continue
        this_similarity = compute_similarity(n1, n2, schema_entities[r], schema_entities[c],
                                             weights[n1][n2])
        matrix[r][c] = this_similarity
        matrix[c][r] = this_similarity
        max_similarity = max(max_similarity, this_similarity)

assert max_similarity > 0.0
# Scale matrix
for r in xrange(no_of_schemas):
    for c in xrange(no_of_schemas):
        matrix[r][c] = matrix[r][c] / max_similarity

if 'similarity_matrix' in args and args['similarity_matrix'] is not None:
    logging.info("Saving the computed similarity matrix to file '%s'...",
                 args['similarity_matrix'].name)
    print("\n".join([ ",".join([ "{0:.6f}".format(el)
                                 for el in mrow])
                      for mrow in matrix ]),
          file=args['similarity_matrix'])

matrixplus = matrix
#matrixminus = [ [ -el for el in mrow ] for mrow in matrix ]
#matrixminus = [ [1]*no_of_schemas for i in xrange(no_of_schemas) ]

n = len(schema_names)
ll = args['min_cardinality']
lu = args['max_cardinality']
eu = args['max_entities']

logging.debug("Clusters must have at least %d and at most %d elements.", ll, lu)
logging.debug("Clusters must have at most %d entities.", eu)

# Compute the number 'k' of clusters

assert 1 <= ll
assert ll <= lu
assert ll <= n

k = int(math.ceil(float(n) / ll))

logging.debug("Set the maximum number of clusters to %d "
              "(due to minimum cardinality constraints).", k)

def lower_triangle_generator(n):
    for i2 in xrange(1, n):
        for i1 in xrange(i2):
            yield (i1, i2)

class VarType:
    (X, Y, Empty, Xtc)= (0, 1, 2, 3)

def prepare_ILP_variables(k, n, matrixplus):
    logging.info("Preparing ILP variables...")
    variable_list= []
    variable_names= []
    variable_obj= []
    variable_types= []
# Variables X (i, j, c) (with i, j elements, and c cluster)
    for i,j in lower_triangle_generator(n):
        variable_list.extend([ (VarType.X, (i, j, c)) for c in xrange(k) ])
        variable_names.extend([ "X_{i}_{j}_{c}".format(i=i, j=j, c=c)
                                for c in xrange(k) ])
        variable_types.extend([ cplex.Cplex.variables.type.continuous ] * k)
        variable_obj.extend([ 0.0 ] * k)

    variable_list.extend(( (VarType.Xtc, c)
                           for c in xrange(k) ))
    variable_names.extend(( "Xtc_{c}".format(c=c)
                           for c in xrange(k) ))
    variable_obj.extend([ 1.0 ] * k)
    variable_types.extend([ cplex.Cplex.variables.type.continuous ] * k)

# Variables Y (i, c) (with i element and c cluster)
    variable_list.extend(( (VarType.Y, (i, c))
                           for i,c in itertools.product(xrange(n), xrange(k)) ))
    variable_names.extend(( "Y_{i}_{c}".format(i=i, c=c)
                           for i,c in itertools.product(xrange(n), xrange(k)) ))
    variable_obj.extend([ 0.0 ] * (n*k))
    variable_types.extend([ cplex.Cplex.variables.type.binary ] * (n*k))

# Variables Empty (c) (with c cluster)
    variable_list.extend(( (VarType.Empty, c)
                           for c in xrange(k) ))
    variable_names.extend(( "Empty_{c}".format(c=c)
                            for c in xrange(k) ))
    variable_obj.extend([ 0.0 ] * k)
    variable_types.extend([ cplex.Cplex.variables.type.binary ] * k)


    variable_idx= { v:i for i,v in enumerate(variable_list) }

    return (variable_names, variable_obj, variable_types, variable_idx)

def prepare_ILP_constraints(k, n, matrixplus,
                            ll, lu, eu,
                            weights,
                            variable_idx):

    def get_constr_coeff(variable_idx, *variables):
        coeff= cplex.SparsePair(ind= [ variable_idx[var] for c,var in variables ],
                                val= [ c for c,var in variables ])
        return coeff

    constr_mat= []
    constr_rhs= []
    constr_sense= []
    # Each element is in a single cluster
    # \sum_k y_{i,k} = 1
    for i in xrange(n):
        constr_mat.append( get_constr_coeff( variable_idx,
                                             *[ (1, (VarType.Y, (i, c)))
                                                for c in xrange(k) ]) )
        constr_sense.append('E')
        constr_rhs.append(1)

    # x's and y's are concordant
    for i,j in lower_triangle_generator(n):
        for c in xrange(k):
            constr_mat.append( get_constr_coeff( variable_idx,
                                                 (matrixplus[i][j], (VarType.Y, (i, c))),
                                                 (-1, (VarType.X, (i, j, c))) ) )
            constr_sense.append('G')
            constr_rhs.append(0)
            constr_mat.append( get_constr_coeff( variable_idx,
                                                 (matrixplus[i][j], (VarType.Y, (j, c))),
                                                 (-1, (VarType.X, (i, j, c))) ) )
            constr_sense.append('G')
            constr_rhs.append(0)

    for c in xrange(k):
        constr_mat.append( get_constr_coeff( variable_idx,
                                             *[(1, (VarType.X, (i, j, c))) for i,j in lower_triangle_generator(n)] +
                                             [ (-1, (VarType.Xtc, c)) ] ) )
        constr_sense.append('G')
        constr_rhs.append(0)

    # Maximum cluster cardinality
    for c in xrange(k):
        constr_mat.append( get_constr_coeff( variable_idx,
                                             *[ (1, (VarType.Y, (i, c)))
                                                for i in xrange(n) ]) )
        constr_sense.append('L')
        constr_rhs.append(lu)

    # Maximum number of entities
    for c in xrange(k):
        constr_mat.append( get_constr_coeff( variable_idx,
                                             *[ (weights[i], (VarType.Y, (i, c)))
                                                for i in xrange(n) ]) )
        constr_sense.append('L')
        constr_rhs.append(eu)

    # Minimum cluster cardinality OR empty
    #   - part 1: Empty_c \ge 1 iff cluster c is empty
    for i,c in itertools.product(xrange(n), xrange(k)):
        constr_mat.append( get_constr_coeff( variable_idx,
                                             (1, (VarType.Empty, c)),
                                             (1, (VarType.Y, (i, c))) ) )
        constr_sense.append('L')
        constr_rhs.append(1)
    #   - part 2: Cardinality of cluster c is at least ll if not Empty_c
    for c in xrange(k):
        constr_mat.append( get_constr_coeff( variable_idx,
                                             *([ (1, (VarType.Y, (i, c)))
                                                 for i in xrange(n) ] +
                                               [ (ll, (VarType.Empty, c)) ]) ) )
        constr_sense.append('G')
        constr_rhs.append(ll)

    # Set an upper bound on the maximum similarity
    n_large_clusters = (n / lu)
    remaining = max(n - (n_large_clusters * lu), ll)
    logging.debug("There are %d large clusters and a single cluster of %d elements.",
                  n_large_clusters, remaining)
    n_elements = (n_large_clusters * (lu*(lu-1)/2)) + (remaining*(remaining-1)/2)
    logging.debug("Computing the %dth largest similarities...", n_elements)
    best_simils = sorted(( matrixplus[i][j] for i,j in lower_triangle_generator(n) ),
                         reverse=True)[:n_elements]
    logging.info("The objective function cannot be greater than %.3f.", sum(best_simils))
    constr_mat.append( get_constr_coeff( variable_idx,
                                         *[ (1, (VarType.X, (i, j, c)))
                                            for i,j in lower_triangle_generator(n)
                                            for c in xrange(k) ] ) )
    constr_sense.append('L')
    constr_rhs.append(sum(best_simils))

    best_cluster_similarity = sum(best_simils[:(lu*(lu-1)/2)])
    logging.debug("Each cluster cannot have similarity greater than %.3f.",
                  best_cluster_similarity)
    for c in xrange(k):
        constr_mat.append( get_constr_coeff( variable_idx,
                                             (1, (VarType.Xtc, c)) ) )
        constr_sense.append('L')
        constr_rhs.append(best_cluster_similarity)




    return (constr_mat, constr_sense, constr_rhs)




logging.debug("Computing the set of variables...")

(variable_names, variable_obj,
 variable_types, variable_idx) = prepare_ILP_variables(k, n, matrixplus)
N_VARS= len(variable_names)

logging.debug("The program has %d variables.", N_VARS)


logging.debug("Computing the set of constraints...")

(constr_mat, constr_sense, constr_rhs) = prepare_ILP_constraints(k, n, matrixplus,
                                                                 ll, lu, eu,
                                                                 [ len(se) for se in schema_entities ],
                                                                 variable_idx)

logging.debug("The program has %d constraints.", len(constr_mat))

c = cplex.Cplex()
with LogFile("CPLEX_LOG == ") as lf:

    c.set_results_stream(lf)
    c.set_log_stream(lf)
    c.set_warning_stream(lf, lambda x: "!! WARNING !! == " + x)
    c.set_error_stream(lf,   lambda x: "!!  ERROR  !! == " + x)

    c.objective.set_sense(c.objective.sense.maximize)

    c.variables.add(names = variable_names,
                    obj = variable_obj,
#                    lb = [ 0.0 ] * N_VARS,
#                    ub = [ 1.0 ] * N_VARS,
                    types = variable_types)



    c.linear_constraints.add(names = [ "c{0}".format(i) for i in range(len(constr_mat)) ],
                             lin_expr = constr_mat,
                             senses = constr_sense,
                             rhs = constr_rhs)

    if args['verbose']>0:
        logging.debug("Saving problem to file 'ilp-problem.mps'...")
        c.write('ilp-problem.mps', 'mps')

    logging.info("Solving the problem...")
    c.solve()

logging.info("Search process terminated!")

STATUS = cplex.Cplex.solution.status
SUBOPTIMAL = [ STATUS.MIP_abort_feasible, STATUS.optimal_tolerance ]
OPTIMAL = [ STATUS.optimal, STATUS.MIP_optimal, STATUS.MIP_optimal_relaxed_sum ]
INFEASIBLE = [ STATUS.infeasible, STATUS.MIP_infeasible, STATUS.MIP_infeasible_or_unbounded ]
logging.debug("The solver status is: %s.", STATUS[c.solution.get_status()])
logging.debug("Solving method: %s.", cplex.Cplex.solution.method[c.solution.get_method()])

if c.solution.get_status() in INFEASIBLE:
    logging.warn("!! No clusterization that satisfies the cardinality constraints is possible !!")
    logging.warn("Given: Min cluster cardinality=%d, Max cluster cardinality=%d, Max no of entities=%d",
                 ll, lu, eu)

if c.solution.get_status() in OPTIMAL or c.solution.get_status() in SUBOPTIMAL:
    if c.solution.get_status() in OPTIMAL:
        logging.info("Optimum found! -- The objective value is %f.",
                     c.solution.get_objective_value())
    else:
        logging.info("Solution found BUT IS NOT GUARANTEED TO BE OPTIMAL! -- The objective value is %f.",
                     c.solution.get_objective_value())

    cluster_assignment = collections.defaultdict(list)
    assigned = set()
    for i in xrange(n):
        for cl in xrange(k):
            if c.solution.get_values(variable_idx[(VarType.Y, (i,cl))]) > 0 :
                assert i not in assigned
                cluster_assignment[cl].append(i)
                assigned.add(i)
                # logging.debug("Schema '%s' is in cluster '%d'.", schema_names[i], cl)

#    for var in variable_names:
#        logging.debug("%.12s= %.5s", var, "True" if c.solution.get_values(var)>0.0 else "False")

    for cl,ls in cluster_assignment.items():
        logging.info("Cluster '%d':", cl)
        for s in ls:
            logging.info("   (%d) %s", s, schema_names[s])
    logging.info("Saving results to file '%s'...", outfile.name)
    print('"CLUSTER-ID","SCHEMA-ID","SCHEMA-NAME"', file=outfile)
    cldiff= 0
    for cl,ls in sorted(cluster_assignment.items(),
                        key=lambda x: x[0]):
        assert not any(( s >= n for s in ls )) or all(( s >= n for s in ls ))
        if any(( s < n for s in ls )):
            for s in ls:
                print('{cl},{sid},"{sname}"'.format(cl=cl-cldiff, sid=s,
                                                    sname=schema_names[s]),
                      file=outfile)
        else:
            cldiff += 1

    logging.debug("Cluster assignment:")
    for cl in xrange(k):
        logging.debug("Cluster {0:3d}: {1}".format(cl,
                                                   "".join([ "X"
                                                             if c.solution.get_values(variable_idx[(VarType.Y, (i,cl))]) > 0
                                                             else "."
                                                             for i,sn in enumerate(schema_names) ])))
    logging.debug("Cluster 'emptiness':")
    for cl in xrange(k):
        logging.debug("Cluster {0:3d}: {1}empty".format(cl, ""
                                                        if c.solution.get_values(variable_idx[(VarType.Empty, cl)]) > 0
                                                        else "not "))

logging.info("Schema integration via clustering -- Completed")




