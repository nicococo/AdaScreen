# -*- coding: utf-8 -*-
# Adapted 2016 Nico Goernitz, TU Berlin
# TU cluster specific changes.

# Written (W) 2008-2012 Christian Widmer
# Written (W) 2008-2010 Cheng Soon Ong
# Written (W) 2012-2014 Daniel Blanchard, dblanchard@ets.org
# Copyright (C) 2008-2012 Max-Planck-Society, 2012-2014 ETS

# This file is part of GridMap.

# GridMap is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# GridMap is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GridMap.  If not, see <http://www.gnu.org/licenses/>.
'''
GridMap provides wrappers that simplify submission and collection of jobs,
in a more 'pythonic' fashion.

:author: Christian Widmer
:author: Cheng Soon Ong
:author: Dan Blanchard (dblanchard@ets.org)

:var USE_MEM_FREE: Does your cluster support specifying how much memory a job
                   will use via mem_free? (Default: ``False``)
:var DEFAULT_QUEUE: The default job scheduling queue to use.
                    (Default: ``all.q``)
:var MAX_TIME_BETWEEN_HEARTBEATS: How long should we wait (in seconds) for a
                                  heartbeat before we consider a job dead?
                                  (Default: 90)
:var IDLE_THRESHOLD: Percent CPU utilization (ratio of CPU time to real time
                     * 100) that a process must drop below to be considered not
                     running.
                     (Default: 1.0)
:var MAX_IDLE_HEARTBEATS: Number of heartbeats we can receive where the process
                          has >= IDLE_THRESHOLD CPU utilization and is sleeping
                          before we consider the process dead. (Default: 3)
:var NUM_RESUBMITS: How many times can a particular job can die, before we give
                    up. (Default: 3)
:var CHECK_FREQUENCY: How many seconds pass before we check on the status of a
                      particular job in seconds. (Default: 15)
:var HEARTBEAT_FREQUENCY: How many seconds pass before jobs on the cluster send
                          back heart beats to the submission host.
                          (Default: 10)
'''

from __future__ import absolute_import, print_function, unicode_literals

from clustermap.conf import (CHECK_FREQUENCY, DEFAULT_QUEUE,
                          HEARTBEAT_FREQUENCY, IDLE_THRESHOLD,
                          MAX_IDLE_HEARTBEATS, MAX_TIME_BETWEEN_HEARTBEATS,
                          NUM_RESUBMITS, USE_MEM_FREE)
from clustermap.job import Job, JobException, process_jobs

# For * imports
__all__ = ['Job', 'JobException', 'process_jobs', 'cluster_map', 'CHECK_FREQUENCY',
           'CREATE_PLOTS', 'DEFAULT_QUEUE', 'HEARTBEAT_FREQUENCY', 'IDLE_THRESHOLD',
           'MAX_IDLE_HEARTBEATS', 'MAX_TIME_BETWEEN_HEARTBEATS', 'NUM_RESUBMITS', 'USE_MEM_FREE']
