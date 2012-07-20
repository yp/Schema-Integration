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
# Module used to describe an abstract MIP solver
#
###


class ProblemType:
    MINIMIZATION = "minimization"
    MAXIMIZATION = "maximization"

    TYPES = (MINIMIZATION, MAXIMIZATION)



class VariableType:
    BINARY = "bin"
    CONTINUOUS = "cont"

    TYPES = (BINARY, CONTINUOUS)



class ConstraintSense:
    LE = "<="
    EQ = "=="
    GE = ">="

    SENSES = (LE, EQ, GE)



class SolutionStatus:
    INFEASIBLE = "infeasible"
    SUBOPTIMAL = "suboptimal"
    OPTIMAL = "optimal"
    INVALID = "invalid"

    STATUSES = (INFEASIBLE, SUBOPTIMAL, OPTIMAL, INVALID)



class Solver:

    def __init__(self, real_solver):
        assert real_solver
        self._real_solver = real_solver
        self._last_status = SolutionStatus.INVALID


    def set_solver_obj(self, problem_type):
        assert problem_type
        assert problem_type in ProblemType.TYPES
        return self._real_solver.set_solver_obj(problem_type)


    def add_variables(self, variables, variable_types,
                      obj_coeff=None, variable_names=None,
                      variable_ranges=None):
        assert variables
        assert variable_types
        assert all((vt in VariableType.TYPES for vt in variable_types))
        assert len(variables) == len(variable_types)
        if not obj_coeff:
            obj_coeff = [ 0.0 ] * len(variables)
        else:
            assert len(obj_coeff) == len(variables)

        return self._real_solver.add_variables(variables, variable_types, obj_coeff,
                                               variable_names, variable_ranges)


    def add_constraint(self, lhs, sense, rhs):
        assert lhs
        assert sense
        assert sense in ConstraintSense.SENSES
        assert type(rhs) in [int, float]
        return self._real_solver.add_constraint(lhs, sense, rhs)


    def add_constraints(self, lhss, senses, rhss):
        return [ self.add_constraints(lhs, sense, rhs)
                 for lhs, sense, rhs in zip(lhss, senses, rhss) ]


    def set_starting_point(self, variable_list, variable_value):
        assert variable_list
        assert variable_value
        assert len(variable_list) == len(variable_value)
        return self._real_solver.set_starting_point(variable_list,
                                                    variable_value)


    def solve(self):
        self._last_status = self._real_solver.solve()
        return self._last_status


    def get_solution_status(self):
        return self._last_status


    def get_value(self, variable):
        assert variable
        return self._real_solver.get_value(variable)

