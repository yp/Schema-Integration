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
# Classes that build/decode an ILP formulation for the problem
#
###

# Standard library
from __future__ import print_function

import collections
import itertools
import logging
import math


# Personal modules
import util
import solution
import solver

class Constraints:
    def __init__(self, ll, lu, eu):
        assert 1 <= ll <= lu
        assert 1 <= eu
        self.ll = ll
        self.lu = lu
        self.eu = eu

    def is_solution_feasible(self, solution):
        if not solution.is_valid():
            return False
        return all(( (self.ll <= len(cluster) <= self.lu) and
                     (sum( (solution.instance.weights[cluster_elem]
                            for cluster_elem in cluster) ) <= self.eu)
                     for cluster in solution.clusters ))

    def __str__(self):
        return ( "Clusters must have at least {0} elements, "
                 "at most {1} elements, "
                 "and at most {2} entities.".format( self.ll,
                                                     self.lu,
                                                     self.eu ) )



class StrictILPFormulation:

    class VarType:
        ClusterAssignment = "Cl"
        Weight = "W"
        WeightCluster = "WCl"
        Empty = "Empty"

    def __init__(self, schemas, constraints):
        assert schemas
        assert constraints
        self._schemas = schemas
        self._constr = constraints
        self._solver = None
        self._elems = None
        self._k = None


    def prepare_ILP(self, ilp_solver, elems=None, k=None):
        vartype = StrictILPFormulation.VarType
        assert ilp_solver
        ## Get all the elements if they are not specified
        if not elems:
            elems = list(range(self._schemas.names))
        ln = len(elems)
        if not k or k == "auto":
            k = int(math.ceil(float(ln) / self._constr.ll))
        kr = list(range(k))

        self._solver = ilp_solver
        self._elems = elems
        self._k = k

        logging.info("Solving [%s] with at most %d clusters...",
                     ", ".join((str(el) for el in self._elems)), self._k)

        self._solver.set_solver_obj(solver.ProblemType.MAXIMIZATION)

        # # # # # # # # # # # # # # #
        # Prepare the VARIABLES
        logging.info("Preparing ILP variables...")
        # - Variables ClusterAssignment (i, c) (with i element and c cluster)
        self._solver.add_variables(
            variables=[ (vartype.ClusterAssignment, (i, c))
                        for i in elems
                        for c in kr ],
            variable_types=[solver.VariableType.BINARY] * (ln*k),
            obj_coeff=[ 0.0 ] * (ln*k) )

        # - Variables Weight (i, j, c) (with i, j elements, and c cluster)
        self._solver.add_variables(
            variables=[ (vartype.Weight, (i1, i2, c))
                        for i1, i2 in util.lower_triangle(elems)
                        for c in kr ],
            variable_types=[solver.VariableType.CONTINUOUS] * (ln*(ln-1)/2*k),
            obj_coeff=[ 0.0 ] * (ln*(ln-1)/2*k) )

        # - Variables WeightCluster (c) (with c cluster)
        self._solver.add_variables(
            variables=[ (vartype.WeightCluster, c) for c in kr ],
            variable_types=[solver.VariableType.CONTINUOUS] * k,
            obj_coeff=[ 1.0 ] * k )

        # - Variables Empty (c) (with c cluster)
        self._solver.add_variables(
            variables=[ (vartype.Empty, c) for c in kr ],
            variable_types=[solver.VariableType.BINARY] * k,
            obj_coeff=[ 0.0 ] * k )


        # # # # # # # # # # # # # # #
        # Prepare the CONSTRAINTS
        logging.info("Preparing ILP constraints...")
        # Each element is in a single cluster
        # \sum_k Cl_{i,k} = 1
        for i in elems:
            self._solver.add_constraint(
                lhs=[ (1, (vartype.ClusterAssignment, (i, c))) for c in kr ],
                sense=solver.ConstraintSense.EQ,
                rhs=1.0 )

        # W_{i,j,k} \le similarity of i and j if i co-clustered with j
        for i1,i2 in util.lower_triangle(elems):
            for c in kr:
                self._solver.add_constraint(
                    lhs=[ (self._schemas.similarities[i1][i2],
                           (vartype.ClusterAssignment, (i1, c))),
                          (-1, (vartype.Weight, (i1, i2, c))) ],
                    sense=solver.ConstraintSense.GE,
                    rhs=0.0 )
                self._solver.add_constraint(
                    lhs=[ (self._schemas.similarities[i1][i2],
                           (vartype.ClusterAssignment, (i2, c))),
                          (-1, (vartype.Weight, (i1, i2, c))) ],
                    sense=solver.ConstraintSense.GE,
                    rhs=0.0 )

        # The weight of a cluster is the sum of the weights
        for c in kr:
            self._solver.add_constraint(
                lhs=( [ (1, (vartype.Weight, (i1, i2, c)))
                        for i1, i2 in util.lower_triangle(elems) ] +
                      [ (-1, (vartype.WeightCluster, c)) ] ),
                sense=solver.ConstraintSense.EQ,
                rhs=0.0 )

        # Maximum cluster cardinality
        for c in kr:
            self._solver.add_constraint(
                lhs=[ (1, (vartype.ClusterAssignment, (i, c)))
                      for i in elems ],
                sense=solver.ConstraintSense.LE,
                rhs=self._constr.lu )

        # Maximum number of entities
        for c in kr:
            self._solver.add_constraint(
                lhs=[ (self._schemas.weights[i],
                       (vartype.ClusterAssignment, (i, c)))
                      for i in elems ],
                sense=solver.ConstraintSense.LE,
                rhs=self._constr.eu )

        # Minimum cluster cardinality OR empty
        #   - part 1: Empty_c \ge 1 iff cluster c is empty
        for i,c in itertools.product(elems, kr):
            self._solver.add_constraint(
                lhs=[ (1, (vartype.Empty, c)),
                      (1, (vartype.ClusterAssignment, (i, c))) ],
                sense=solver.ConstraintSense.LE,
                rhs=1 )
        #   - part 2: Cardinality of cluster c is at least ll if not Empty_c
        for c in kr:
            self._solver.add_constraint(
                lhs=( [ (1, (vartype.ClusterAssignment, (i, c))) for i in elems ] +
                      [ (self._constr.ll, (vartype.Empty, c)) ] ),
                sense=solver.ConstraintSense.GE,
                rhs=self._constr.ll )

        '''
    # Cluster similarities must be sorted
    for c1,c2 in consecutive_pairs_generator(kr):
        constr_mat.append( get_constr_coeff( variable_idx,
                                             ( 1, (vartype.WeightCluster, c1)),
                                             (-1, (vartype.WeightCluster, c2)) ) )
        constr_sense.append('G')
        constr_rhs.append(0)
    # Cluster emptiness must be sorted
    for c1,c2 in consecutive_pairs_generator(kr):
        constr_mat.append( get_constr_coeff( variable_idx,
                                             ( 1, (vartype.Empty, c2)),
                                             (-1, (vartype.Empty, c1)) ) )
        constr_sense.append('G')
        constr_rhs.append(0)

    n_large_clusters = (lnr / lu)
    remaining = lnr - (n_large_clusters * lu)
    if 0 < remaining and remaining < ll:
        remaining = ll
    logging.debug("There are %d large clusters and a single cluster of %d elements.",
                  n_large_clusters, remaining)
    n_elements = (n_large_clusters * (lu*(lu-1)/2)) + (remaining*(remaining-1)/2)
    logging.debug("Computing the %dth largest similarities...", n_elements)
    simils = sorted(( matrixplus[i][j] for i,j in lower_triangle_generator(nr) ),
                    reverse=True)
    logging.info("The objective function cannot be greater than %.3f.",
                 sum(simils[:n_elements]))
    constr_mat.append( get_constr_coeff( variable_idx,
                                         *[ (1, (vartype.WeightCluster, c))
                                            for c in kr ] ) )
    constr_sense.append('L')
    constr_rhs.append(sum(simils[:n_elements]))

    simils = [x for i,row in [ (el,matrixplus[el]) for el in nr ]
              for x in sorted([row[j] for j in nr if j < el], reverse=True)[:lu]]
    other_obj_ub = sum(sorted(simils, reverse=True)[:n_elements])
    constr_mat.append( get_constr_coeff( variable_idx,
                                         *[ (1, (vartype.WeightCluster, c))
                                            for c in kr ] ) )
    constr_sense.append('L')
    constr_rhs.append(other_obj_ub)
    logging.info("The objective function cannot be greater than %.3f.",
                 other_obj_ub)
                 '''

    def get_solution(self):
        vartype = StrictILPFormulation.VarType
        assert self._solver
        solv = self._solver
        ss = self._solver.get_solution_status()
        if ss in [solver.SolutionStatus.INFEASIBLE, solver.SolutionStatus.INVALID]:
            logging.warn("The solver has not a valid solution. Solver status: '%s'", ss)
            return (ss, None)
        # Optimal or suboptimal
        logging.info("The solver computed a *%s* solution.", ss)

        # Get back the solution
        clusters = [ [ i  for i in self._elems
                       if solv.get_value((vartype.ClusterAssignment, (i, c))) > 0.0 ]
                     for c in range(self._k) ]
        clusters = [ cluster for cluster in clusters if cluster ]

        # Log the solution
        logging.debug("Computed solution:")
        for cluster_id,cluster in enumerate(clusters):
            logging.debug("Cl #%-3d (%3d): %s", cluster_id, len(cluster),
                          "".join(( "X" if el in cluster else "."
                                    for el in self._elems )) )

        # Check solution validity
        in_clusters = [ el for cluster in clusters for el in cluster ]
        is_valid = ( all(( el in self._elems for el in in_clusters )) and
                     len(in_clusters) == len(set(in_clusters)) )

        logging.debug("The cluster assignment is %svalid.",
                      "" if is_valid else "not ")

        if not is_valid:
            logging.warn("The solution computed by the solver is not valid.")
            ss = solver.SolutionStatus.INVALID
            return (ss, None)

        # Build the solution
        s = solution.Solution(self._schemas, clusters)
        logging.debug("The solution is %scomplete.",
                      "" if s.is_complete() else "not ")

        if not self._constr.is_solution_feasible(s):
            logging.warn("The solution is valid but NOT feasible "
                         "given the constraints '%s'.", self._constr)
            ss = solver.SolutionStatus.INVALID
        else:
            logging.debug("The solution is feasible.")

        return (ss, s)
