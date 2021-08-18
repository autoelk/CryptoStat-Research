//******************************************************************************
//
// File:    MinVerCovSeq3.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.MinVerCovSeq3
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

import edu.rit.pj2.Task;
import edu.rit.util.BitSet;
import edu.rit.util.GraphSpec;
import edu.rit.util.Instance;
import edu.rit.util.IntAction;
import edu.rit.util.Random;
import edu.rit.util.RandomSubset;

/**
 * Class MinVerCovSeq3 is a sequential program that finds a minimum vertex cover
 * of a graph via heuristic search. The program constructs a {@linkplain
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
 * Usage: <TT>java pj2 edu.rit.pj2.example.MinVerCovSeq3 "<I>ctor</I>"
 * <I>seed</I> <I>N</I></TT>
 * <BR><TT><I>ctor</I></TT> = GraphSpec constructor expression
 * <BR><TT><I>seed</I></TT> = Random seed
 * <BR><TT><I>N</I></TT> = Number of trials
 *
 * @author  Alan Kaminsky
 * @version 26-Apr-2018
 */
public class MinVerCovSeq3
	extends Task
	{
	// Graph being analyzed.
	AMGraph graph;
	int V;

	// Minimum vertex cover.
	BitSet minCover;

	// Candidate vertex set.
	BitSet candidate;

	/**
	 * Main program.
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

		// Construct graph spec, set up adjacency matrix.
		graph = new AMGraph ((GraphSpec) Instance.newInstance (ctor));
		V = graph.V();

		// Set up pseudorandom number generator and random subset generator.
		Random prng = new Random (seed);
		RandomSubset rsg = new RandomSubset (prng, V, true);

		// Check N randomly chosen candidate covers.
		minCover = new BitSet (V) .add (0, V);
		candidate = new BitSet (V);
		for (long i = 0L; i < N; ++ i)
			{
			candidate.clear();
			rsg.restart();
			while (! graph.isVertexCover (candidate))
				candidate.add (rsg.next());
			if (candidate.size() < minCover.size())
				minCover.copy (candidate);
			}

		// Print results.
		System.out.printf ("Cover =");
		minCover.forEachItemDo (new IntAction()
			{
			public void run (int i)
				{
				System.out.printf (" %d", i);
				}
			});
		System.out.println();
		System.out.printf ("Size = %d%n", minCover.size());
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.MinVerCovSeq3 \"<ctor>\" <seed> <N>");
		System.err.println ("<ctor> = GraphSpec constructor expression");
		System.err.println ("<seed> = Random seed");
		System.err.println ("<N> = Number of trials");
		terminate (1);
		}

	/**
	 * Specify that this task requires one core.
	 */
	protected static int coresRequired()
		{
		return 1;
		}

	}
