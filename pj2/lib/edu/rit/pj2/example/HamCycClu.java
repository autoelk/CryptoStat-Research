//******************************************************************************
//
// File:    HamCycClu.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.HamCycClu
//
// This Java source file is copyright (C) 2018 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This Java source file is part of the Parallel Java 2 Library ("PJ2"). PJ2 is
// free software; you can redistribute it and/or modify it under the terms of
// the GNU General Public License as published by the Free Software Foundation;
// either version 3 of the License, or (at your option) any later version.
//
// PJ2 is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************

package edu.rit.pj2.example;

import edu.rit.pj2.Job;
import edu.rit.pj2.JobWorkQueue;
import edu.rit.pj2.Task;
import edu.rit.pj2.TupleListener;
import edu.rit.pj2.tuple.ObjectTuple;
import edu.rit.util.GraphSpec;
import edu.rit.util.Instance;

/**
 * Class HamCycClu is a cluster parallel program that finds a Hamiltonian cycle
 * in a graph via exhaustive search. The program constructs a {@linkplain
 * GraphSpec GraphSpec} object using the constructor expression on the command
 * line, then uses the graph spec to construct the graph.
 * <P>
 * For further information about constructor expressions, see class {@linkplain
 * edu.rit.util.Instance edu.rit.util.Instance}.
 * <P>
 * Usage: <TT>java pj2 [workers=<I>K</I>] edu.rit.pj2.example.HamCycClu
 * "<I>ctor</I>" <I>threshold</I></TT>
 * <BR><TT><I>K</I></TT> = Number of worker tasks (default 1)
 * <BR><TT><I>ctor</I></TT> = GraphSpec constructor expression
 * <BR><TT><I>threshold</I></TT> = Parallel search threshold level
 * <P>
 * The program traverses the exhaustive search tree down to the given
 * <I>threshold</I> level in a breadth first fashion. The program then searches
 * the subtrees at that level in parallel in a depth first fashion. The
 * <I>threshold</I> should be specified so there are enough subtrees to balance
 * the load among the cluster nodes and cores.
 *
 * @author  Alan Kaminsky
 * @version 01-May-2018
 */
public class HamCycClu
	extends Job
	{

	/**
	 * Job main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Parse command line arguments.
		if (args.length != 2) usage();
		String ctor = args[0];
		int threshold = Integer.parseInt (args[1]);

		// Set up job's distributed work queue.
		JobWorkQueue<HamCycState> queue =
			getJobWorkQueue (HamCycState.class, workers());

		// Construct graph spec, set up graph.
		HamCycStateClu.setGraph
			(new AMGraph ((GraphSpec) Instance.newInstance (ctor)),
			 threshold, queue);

		// Add first work item to work queue.
		queue.add (new HamCycStateClu());

		// Set up team of worker tasks.
		rule() .task (workers(), SearchTask.class) .args (ctor, ""+threshold);

		// Set up task to print results.
		rule() .atFinish() .task (ResultTask.class) .runInJobProcess();
		}

	/**
	 * Search task.
	 */
	private static class SearchTask
		extends Task
		{
		// Search task main program.
		public void main
			(String[] args)
			throws Exception
			{
			// Parse command line arguments.
			String ctor = args[0];
			int threshold = Integer.parseInt (args[1]);

			// Set up job's distributed work queue.
			JobWorkQueue<HamCycState> queue =
				getJobWorkQueue (HamCycState.class);

			// Construct graph spec, set up graph.
			HamCycStateClu.setGraph
				(new AMGraph ((GraphSpec) Instance.newInstance (ctor)),
				 threshold, queue);

			// Stop the search when any task finds a Hamiltonian cycle.
			addTupleListener (new TupleListener<ObjectTuple<HamCycState>>
				(new ObjectTuple<HamCycState>())
				{
				public void run (ObjectTuple<HamCycState> tuple)
					{
					HamCycState.stop();
					}
				});

			// Search for a Hamiltonian cycle.
			HamCycState state;
			while ((state = queue.remove()) != null)
				{
				HamCycState cycle = state.search();
				if (cycle != null)
					putTuple (new ObjectTuple<HamCycState> (cycle));
				}
			}

		// The search task requires one core.
		protected static int coresRequired()
			{
			return 1;
			}
		}

	/**
	 * Result task.
	 */
	private static class ResultTask
		extends Task
		{
		// Task main program.
		public void main
			(String[] args)
			throws Exception
			{
			ObjectTuple<HamCycState> template = new ObjectTuple<HamCycState>();
			ObjectTuple<HamCycState> cycle = tryToReadTuple (template);
			if (cycle != null)
				System.out.println (cycle.item);
			else
				System.out.println ("None");
			}
		}

	/**
	 * Print an error message and exit.
	 */
	private static void usage
		(String msg)
		{
		System.err.printf ("HamCycClu: %s%n", msg);
		usage();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 [workers=<K>] edu.rit.pj2.example.HamCycClu \"<ctor>\" <threshold>");
		System.err.println ("<K> = Number of worker tasks (default 1)");
		System.err.println ("<ctor> = GraphSpec constructor expression");
		System.err.println ("<threshold> = Parallel search threshold level");
		terminate (1);
		}

	}
