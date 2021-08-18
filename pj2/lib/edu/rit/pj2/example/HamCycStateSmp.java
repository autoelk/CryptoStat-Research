//******************************************************************************
//
// File:    HamCycStateSmp.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.HamCycStateSmp
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

import edu.rit.pj2.WorkQueue;

/**
 * Class HamCycStateSmp encapsulates the state of a search for a Hamiltonian
 * cycle in a graph in a multicore parallel program. Class HamCycStateSmp
 * supports both breadth first search (BFS) and depth first search (DFS).
 *
 * @author  Alan Kaminsky
 * @version 30-Apr-2018
 */
public class HamCycStateSmp
	extends HamCycState
	{

// Hidden class data members.

	// Multicore parallel work queue.
	private static WorkQueue<HamCycState> queue;

// Exported class operations.

	/**
	 * Specify the graph to be analyzed using BFS and DFS.
	 *
	 * @param  graph      Graph.
	 * @param  threshold  Search level threshold at which to switch from BFS to
	 *                    DFS.
	 * @param  queue      Multicore parallel work queue.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>graph</TT> is null. Thrown if
	 *     <TT>queue</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>threshold</TT> &lt; 0.
	 */
	public static void setGraph
		(AMGraph graph,
		 int threshold,
		 WorkQueue<HamCycState> queue)
		{
		if (queue == null)
			throw new NullPointerException
				("HamCycStateSmp.setGraph(): queue is null");
		HamCycState.setGraph (graph);
		HamCycState.threshold = threshold;
		HamCycStateSmp.queue = queue;
		}

// Exported constructors.

	/**
	 * Construct a new search state object that supports BFS and DFS. All
	 * instances of class HamCycStateSmp will analyze the graph specified in the
	 * {@link #setGraph(AMGraph,int,WorkQueue) setGraph()} method.
	 */
	public HamCycStateSmp()
		{
		super();
		}

// Hidden operations.

	/**
	 * Enqueue the given search state object during a BFS.
	 *
	 * @param  state  Search state object to enqueue.
	 */
	protected void enqueue
		(HamCycState state)
		{
		queue.add (state);
		}

	}
