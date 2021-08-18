//******************************************************************************
//
// File:    MinVerCovClu3.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.MinVerCovClu3
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
import edu.rit.pj2.LongLoop;
import edu.rit.pj2.Task;
import edu.rit.pj2.vbl.BitSetVbl;
import edu.rit.util.BitSet;
import edu.rit.util.GraphSpec;
import edu.rit.util.Instance;
import edu.rit.util.IntAction;
import edu.rit.util.Random;
import edu.rit.util.RandomSubset;

/**
 * Class MinVerCovSeq3 is a cluster parallel program that finds a minimum vertex
 * cover of a graph via heuristic search. The program constructs a {@linkplain
 * GraphSpec GraphSpec} object using the constructor expression on the command
 * line, then uses the graph spec to construct the graph.
 * <P>
 * The program performs <I>N</I> trials. For each trial, the program starts with
 * an empty vertex set and adds vertices chosen at random until the vertex set
 * is a cover. The program reports the smallest cover found.
 * <P>
 * For further information about constructor expressions, see class {@linkplain
 * edu.rit.util.Instance edu.rit.util.Instance}.
 * <P>
 * Usage: <TT>java pj2 [workers=<I>K</I>] edu.rit.pj2.example.MinVerCovClu3
 * \"<I>ctor</I>\" <I>seed</I> <I>N</I></TT>
 * <BR><TT><I>K</I></TT> = Number of worker tasks (default 1)
 * <BR><TT><I>ctor</I></TT> = GraphSpec constructor expression
 * <BR><TT><I>seed</I></TT> = Random seed
 * <BR><TT><I>N</I></TT> = Number of trials
 *
 * @author  Alan Kaminsky
 * @version 26-Apr-2018
 */
public class MinVerCovClu3
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
		if (args.length != 3) usage();
		String ctor = args[0];
		long seed = Long.parseLong (args[1]);
		long N = Long.parseLong (args[2]);

		// Set up a task group of K worker tasks.
		masterFor (0, N - 1, WorkerTask.class) .args (ctor, ""+seed);

		// Set up reduction task.
		rule() .atFinish() .task (ReduceTask.class) .runInJobProcess();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 [workers=<K>] edu.rit.pj2.example.MinVerCovClu3 \"<ctor>\" <seed> <N>");
		System.err.println ("<K> = Number of worker tasks (default 1)");
		System.err.println ("<ctor> = GraphSpec constructor expression");
		System.err.println ("<seed> = Random seed");
		System.err.println ("<N> = Number of trials");
		terminate (1);
		}

	/**
	 * Worker task class.
	 */
	private static class WorkerTask
		extends Task
		{
		// Random seed.
		long seed;

		// Graph being analyzed.
		AMGraph graph;
		int V;

		// Minimum vertex cover.
		BitSetVbl minCover;

		/**
		 * Worker task main program.
		 */
		public void main
			(String[] args)
			throws Exception
			{
			// Parse command line arguments.
			String ctor = args[0];
			seed = Long.parseLong (args[1]);

			// Set up adjacency matrix.
			graph = new AMGraph ((GraphSpec) Instance.newInstance (ctor));
			V = graph.V();

			// Check randomly chosen candidate covers.
			minCover = new BitSetVbl.MinSize (new BitSet (V));
			minCover.bitset.add (0, V);
			workerFor() .exec (new LongLoop()
				{
				BitSetVbl thrMinCover;
				BitSet candidate;
				Random prng;
				RandomSubset rsg;
				public void start()
					{
					thrMinCover = threadLocal (minCover);
					candidate = new BitSet (V);
					prng = new Random (seed + 1000*taskRank() + rank());
					rsg = new RandomSubset (prng, V, true);
					}
				public void run (long i)
					{
					candidate.clear();
					rsg.restart();
					while (! graph.isVertexCover (candidate))
						candidate.add (rsg.next());
					if (candidate.size() < thrMinCover.bitset.size())
						thrMinCover.bitset.copy (candidate);
					}
				});

			// Send best candidate cover to reduction task.
			putTuple (minCover);
			}
		}

	/**
	 * Reduction task class.
	 */
	private static class ReduceTask
		extends Task
		{
		/**
		 * Reduction task main program.
		 */
		public void main
			(String[] args)
			throws Exception
			{
			// Reduce all worker task results together.
			BitSetVbl template = new BitSetVbl();
			BitSetVbl minCover = takeTuple (template);
			BitSetVbl taskCover;
			while ((taskCover = tryToTakeTuple (template)) != null)
				minCover.reduce (taskCover);

			// Print final result.
			System.out.printf ("Cover =");
			minCover.bitset.forEachItemDo (new IntAction()
				{
				public void run (int i)
					{
					System.out.printf (" %d", i);
					}
				});
			System.out.println();
			System.out.printf ("Size = %d%n", minCover.bitset.size());
			}
		}

	}
