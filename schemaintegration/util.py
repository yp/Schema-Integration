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
# Some utility functions
#
###

# Standard library
import collections


def lower_triangle(n):
    v = None
    if not isinstance(n, collections.Iterable):
        v = list(range(n))
    else:
        v = list(n)

    for i2 in xrange(1, len(v)):
        for i1 in xrange(i2):
            yield (v[i1], v[i2])




def consecutive_pairs(n):
    if n:
        v = None
        if not isinstance(n, collections.Iterable):
            v = list(range(n))
        else:
            v = list(n)

        prev = v[0]
        for i in xrange(1, len(v)):
            newv = v[i]
            yield (prev, newv)
            prev = newv




def positive_integer(string):
    value = None
    try:
        value = int(string)
    except:
        raise argparse.ArgumentTypeError("'{0}' is not a positive integer.".format(string))
    if value <= 0:
        raise argparse.ArgumentTypeError("'{0}' is not a positive integer.".format(string))
    return value



def fmt_var(variable):
    if type(variable) is tuple:
        return "_".join([str(part) for part in variable])
    else:
        return str(variable)


