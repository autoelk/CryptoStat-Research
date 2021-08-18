//******************************************************************************
//
// File:    MinVerCovSmp2.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.MinVerCovSmp2
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

import edu.rit.pj2.LongLoop;
import edu.rit.pj2.Task;
import edu.rit.pj2.vbl.BitSet64Vbl;
import edu.rit.util.BitSet64;
import edu.rit.util.GraphSpec;
import edu.rit.util.Instance;
import edu.rit.util.IntAction;

/**
 * Class MinVerCovSmp2 is an SMP parallel program that finds a minimum vertex
 * cover of a graph via exhaustive search. The program constructs a {@linkplain
 * GraphSpec GraphSpec} object using the constructor expression on the command
 * line, then uses the graph spec to construct the graph. The program's running
 * time is proportional to 2<SUP><I>V</I></SUP>, where <I>V</I> is the number of
 * vertices in the graph.
 * <P>
 * Class {@linkplain MinVerCovSmp} constructs a new object on every loop
 * iteration. Class MinVerCovSmp2 reuses the same object on every loop
 * iteration.
 * <P>
 * For further information about constructor expressions, see class {@linkplain
 * edu.rit.util.Instance edu.rit.util.Instance}.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.example.MinVerCovSmp2 "<I>ctor</I>"</TT>
 * <BR><TT><I>ctor</I></TT> = GraphSpec constructor expression
 *
 * @author  Alan Kaminsky
 * @version 26-Apr-2018
 */
public class MinVerCovSmp2
	extends Task
	{
	// Graph being analyzed.
	AMGraph64 graph;
	int V;

	// Minimum vertex cover.
	BitSet64Vbl minCover;

	/**
	 * Main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Parse command line arguments.
		if (args.length != 1) usage();
		String ctor = args[0];

		// Construct graph spec, set up adjacency matrix.
		graph = new AMGraph64 ((GraphSpec) Instance.newInstance (ctor));
		V = graph.V();
		if (V > 63) error ("Too many vertices");

		// Check all candidate covers (sets of vertices).
		minCover = new BitSet64Vbl.MinSize();
		minCover.bitset.add (0, V);
		long full = minCover.bitset.bitmap();
		parallelFor (0L, full) .exec (new LongLoop()
			{
			BitSet64Vbl thrMinCover;
			BitSet64 candidate;
			public void start()
				{
				thrMinCover = threadLocal (minCover);
				candidate = new BitSet64();
				}
			public void run (long elems)
				{
				candidate.bitmap (elems);
				if (candidate.size() < thrMinCover.bitset.size() &&
					graph.isVertexCover (candidate))
						thrMinCover.bitset.copy (candidate);
				}
			});

		// Print results.
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

	/**
	 * Print an error message and exit.
	 */
	private static void error
		(String msg)
		{
		System.err.printf ("MinVerCovSmp2: %s%n", msg);
		terminate (1);
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.MinVerCovSmp2 \"<ctor>\"");
		System.err.println ("<ctor> = GraphSpec constructor expression");
		terminate (1);
		}

	}
