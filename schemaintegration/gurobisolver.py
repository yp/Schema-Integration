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
# Interface to the GUROBI solver
#
###

# Standard library
from __future__ import print_function

import collections
import itertools
import logging

import gurobipy


# Personal modules
import util
import solver


class GurobiSolver:

    STATUS = gurobipy.GRB
    SUBOPTIMAL = [ STATUS.ITERATION_LIMIT, STATUS.NODE_LIMIT, STATUS.TIME_LIMIT,
                   STATUS.SOLUTION_LIMIT, STATUS.INTERRUPTED, STATUS.SUBOPTIMAL ]
    OPTIMAL = [ STATUS.OPTIMAL ]
    INFEASIBLE = [ STATUS.INFEASIBLE, STATUS.INF_OR_UNBD, STATUS.UNBOUNDED ]

    TYPE_MAPPING = { solver.VariableType.BINARY: gurobipy.GRB.BINARY,
                     solver.VariableType.CONTINUOUS: gurobipy.GRB.CONTINUOUS }

    SENSE_MAPPING = { solver.ConstraintSense.LE: gurobipy.GRB.LESS_EQUAL,
                      solver.ConstraintSense.EQ: gurobipy.GRB.EQUAL,
                      solver.ConstraintSense.GE: gurobipy.GRB.GREATER_EQUAL }

    def __init__(self):
        logging.info("Using GUROBI as MIP solver...")
        self._m= gurobipy.Model("model")
        self._m.setParam('threads', 2)

        self._variable_idx = {}
        self._objs = []
        self._res_status = solver.SolutionStatus.INVALID
        self._problem_type = None


    def set_solver_obj(self, problem_type):
        self._problem_type = problem_type
        return None


    def add_variables(self, variables, variable_types,
                      obj_coeff, variable_names, variable_ranges):
        assert all((v not in self._variable_idx for v in variables))
        if not variable_names:
            variable_names = [ util.fmt_var(v) for v in variables ]
        else:
            assert len(variable_names) == len(variables)
        if not variable_ranges:
            variable_lb = [ 0.0 ] * len(variables)
            variable_ub = [ gurobipy.GRB.INFINITY ] * len(variables)
        else:
            assert len(variable_ranges) == len(variables)
            variable_lb = [ r[0] for r in variable_ranges ]
            variable_ub = [ r[1] for r in variable_ranges ]
        for i,v in enumerate(variables):
            self._variable_idx[v] = self._m.addVar(
                lb=    variable_lb[i],
                ub=    variable_ub[i],
                vtype= GurobiSolver.TYPE_MAPPING[variable_types[i]],
                name=  variable_names[i])
            if obj_coeff[i] > 0.0 or obj_coeff[i] < 0.0:
                self._objs.append((obj_coeff[i], self._variable_idx[v]))
        self._m.update()
        return None


    def add_constraint(self, lhs, sense, rhs):
        assert all((v in self._variable_idx for c,v in lhs))
        gexpr = gurobipy.LinExpr([ (c, self._variable_idx[v])
                                   for c,v in lhs ])
        self._m.addConstr(lhs=   gexpr,
                          sense= GurobiSolver.SENSE_MAPPING[sense],
                          rhs=   rhs)
        return None


    def set_starting_point(self, variable_list, variable_value):
        logging.warn("GurobiSolver.set_starting_point(...) is NOT IMPLEMENTED!!")
        return None


    def solve(self):
        self._res_status = solver.SolutionStatus.INVALID
        self._m.update()
        if self._problem_type == solver.ProblemType.MINIMIZATION:
            self._m.setObjective(gurobipy.LinExpr(self._objs), gurobipy.GRB.MINIMIZE)
        else:
            self._m.setObjective(gurobipy.LinExpr(self._objs), gurobipy.GRB.MAXIMIZE)
        self._m.update()
        logging.debug("Saving problem to file 'ilp-problem.mps.gz'...")
        self._m.write('gurobi-ilp-problem.mps.gz')

        logging.info("Starting optimization with Gurobi...")
        self._m.optimize()
        logging.info("Optimization terminated.")

        status = self._m.status
        logging.info("The solver status is: %s.", status)

        if status in GurobiSolver.OPTIMAL:
            self._res_status = solver.SolutionStatus.OPTIMAL
        elif status in GurobiSolver.SUBOPTIMAL:
            self._res_status = solver.SolutionStatus.SUBOPTIMAL
        elif status in GurobiSolver.INFEASIBLE:
            self._res_status = solver.SolutionStatus.INFEASIBLE

        return self._res_status


    def get_value(self, variable):
        assert self._res_status != solver.SolutionStatus.INVALID
        assert variable in self._variable_idx
        return self._variable_idx[variable].x
