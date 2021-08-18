//******************************************************************************
//
// File:    HamCycSmp.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.HamCycSmp
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

import edu.rit.pj2.ObjectLoop;
import edu.rit.pj2.Task;
import edu.rit.pj2.WorkQueue;
import edu.rit.util.GraphSpec;
import edu.rit.util.Instance;

/**
 * Class HamCycSmp is a multicore parallel program that finds a Hamiltonian
 * cycle in a graph via exhaustive search. The program constructs a {@linkplain
 * GraphSpec GraphSpec} object using the constructor expression on the command
 * line, then uses the graph spec to construct the graph.
 * <P>
 * For further information about constructor expressions, see class {@linkplain
 * edu.rit.util.Instance edu.rit.util.Instance}.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.example.HamCycSmp "<I>ctor</I>"
 * <I>threshold</I></TT>
 * <BR><TT><I>ctor</I></TT> = GraphSpec constructor expression
 * <BR><TT><I>threshold</I></TT> = Parallel search threshold level
 * <P>
 * The program traverses the exhaustive search tree down to the given
 * <I>threshold</I> level in a breadth first fashion. The program then searches
 * the subtrees at that level in parallel in a depth first fashion. The
 * <I>threshold</I> should be specified so there are enough subtrees to balance
 * the load among the parallel threads.
 *
 * @author  Alan Kaminsky
 * @version 26-Apr-2018
 */
public class HamCycSmp
	extends Task
	{
	// Hamiltonian cycle that was found.
	HamCycState hamCycle;

	/**
	 * Task main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Parse command line arguments.
		if (args.length != 2) usage();
		String ctor = args[0];
		int threshold = Integer.parseInt (args[1]);

		// Set up parallel work queue.
		WorkQueue<HamCycState> queue = new WorkQueue<HamCycState>();

		// Construct graph spec, set up graph.
		HamCycStateSmp.setGraph
			(new AMGraph ((GraphSpec) Instance.newInstance (ctor)),
			 threshold, queue);

		// Add first work item to work queue.
		queue.add (new HamCycStateSmp());

		// Search the graph in parallel.
		parallelFor (queue) .exec (new ObjectLoop<HamCycState>()
			{
			public void run (HamCycState state)
				{
				HamCycState cycle = state.search();
				if (cycle != null)
					{
					HamCycState.stop();
					hamCycle = cycle;
					}
				}
			});

		// Print results.
		if (hamCycle != null)
			System.out.println (hamCycle);
		else
			System.out.println ("None");
		}

	/**
	 * Print an error message and exit.
	 */
	private static void usage
		(String msg)
		{
		System.err.printf ("HamCycSmp: %s%n", msg);
		usage();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.HamCycSmp \"<ctor>\" <threshold>");
		System.err.println ("<ctor> = GraphSpec constructor expression");
		System.err.println ("<threshold> = Parallel search threshold level");
		terminate (1);
		}

	}
