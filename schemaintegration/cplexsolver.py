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
# Interface to the CPLEX solver
#
###

# Standard library
from __future__ import print_function

import collections
import itertools
import logging

import cplex


# Personal modules
import util
import solver


class LogFile:
    def __init__(self, prefix="", log_function=logging.debug):
        assert log_function
        self._buff = []
        self._prefix = prefix
        self._logfn = log_function

    def _do_print(self):
        self._logfn("%s%s",
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


class CplexSolver:

    STATUS = cplex.Cplex.solution.status
    SUBOPTIMAL = [ STATUS.MIP_abort_feasible, STATUS.optimal_tolerance ]
    OPTIMAL = [ STATUS.optimal, STATUS.MIP_optimal, STATUS.MIP_optimal_relaxed_sum ]
    INFEASIBLE = [ STATUS.infeasible, STATUS.MIP_infeasible, STATUS.MIP_infeasible_or_unbounded ]

    TYPE_MAPPING = { solver.VariableType.BINARY: cplex.Cplex.variables.type.binary,
                     solver.VariableType.CONTINUOUS: cplex.Cplex.variables.type.continuous }

    SENSE_MAPPING = { solver.ConstraintSense.LE: "L",
                      solver.ConstraintSense.EQ: "E",
                      solver.ConstraintSense.GE: "G" }

    def __init__(self):
        logging.info("Using ILOG Cplex as MIP solver...")
        self._c= cplex.Cplex()
        self._c.parameters.threads.set(2)
        self._c.parameters.emphasis.memory.set(1)
        self._c.parameters.emphasis.mip.set(1)
        self._c.parameters.parallel.set(0)
        self._c.parameters.mip.strategy.search.set(1)
        self._c.parameters.mip.strategy.probe.set(3)

        self._variable_idx = collections.defaultdict(itertools.count(0).next)
        self._res_status = solver.SolutionStatus.INVALID


    def set_solver_obj(self, problem_type):
        if problem_type == solver.ProblemType.MINIMIZATION:
            self._c.objective.set_sense(cplex.Cplex.objective.sense.minimize)
        else:
            self._c.objective.set_sense(cplex.Cplex.objective.sense.maximize)
        return None


    def add_variables(self, variables, variable_types,
                      obj_coeff, variable_names, variable_ranges):
        assert all((v not in self._variable_idx for v in variables))
        if not variable_names:
            variable_names = [ util.fmt_var(v) for v in variables ]
        else:
            assert len(variable_names) == len(variables)
        if not variable_ranges:
            variable_lb = []
            variable_ub = []
        else:
            assert len(variable_ranges) == len(variables)
            variable_lb = [ r[0] for r in variable_ranges ]
            variable_ub = [ r[1] for r in variable_ranges ]
        [ self._variable_idx[v] for v in variables ]
        self._c.variables.add(obj=   obj_coeff,
                              types= [ CplexSolver.TYPE_MAPPING[vt] for vt in variable_types ],
                              lb=    variable_lb,
                              ub=    variable_ub,
                              names= variable_names)
        return None


    def add_constraint(self, lhs, sense, rhs):
        def get_linear_expression(expr):
            coeff= cplex.SparsePair(ind= [ self._variable_idx[v] for c,v in expr ],
                                    val= [ c for c,v in expr ])
            return coeff

        assert all((v in self._variable_idx for c,v in lhs))
        self._c.linear_constraints.add(lin_expr = [get_linear_expression(lhs)],
                                       senses = [CplexSolver.SENSE_MAPPING[sense]],
                                       rhs = [rhs])
        return None


    def set_starting_point(self, variable_list, variable_value):
        logging.warn("CplexSolver.set_starting_point(...) is NOT IMPLEMENTED!!")
        return None


    def solve(self):
        self._res_status = solver.SolutionStatus.INVALID
        with LogFile("CPLEX_LOG == ") as lf:
            self._c.set_results_stream(lf)
            self._c.set_log_stream(lf)
            self._c.set_warning_stream(lf, lambda x: "!! WARNING !! == " + x)
            self._c.set_error_stream(lf,   lambda x: "!!  ERROR  !! == " + x)

            logging.debug("Saving problem to file 'ilp-problem.mps'...")
            self._c.write('ilp-problem.mps', 'mps')

            logging.info("Starting optimization with CPLEX...")
            self._c.solve()
            logging.info("Optimization terminated.")

            status = self._c.solution.get_status()
            logging.info("The solver status is: %s.", CplexSolver.STATUS[status])

            if status in CplexSolver.OPTIMAL:
                self._res_status = solver.SolutionStatus.OPTIMAL
            elif status in CplexSolver.SUBOPTIMAL:
                self._res_status = solver.SolutionStatus.SUBOPTIMAL
            elif status in CplexSolver.INFEASIBLE:
                self._res_status = solver.SolutionStatus.INFEASIBLE

        return self._res_status


    def get_value(self, variable):
        assert self._res_status != solver.SolutionStatus.INVALID
        assert variable in self._variable_idx
        return self._c.solution.get_values(self._variable_idx[variable])
