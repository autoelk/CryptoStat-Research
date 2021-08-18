//******************************************************************************
//
// File:    HamCycState.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.HamCycState
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

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.pj2.Tuple;
import java.io.IOException;
import java.util.Formatter;

/**
 * Class HamCycState encapsulates the state of a search for a Hamiltonian cycle
 * in a graph.
 * <P>
 * The HamCycState base class supports depth first search (DFS) in a sequential
 * program. Subclasses support breadth first search (BFS) and DFS in parallel
 * programs.
 *
 * @author  Alan Kaminsky
 * @version 30-Apr-2018
 */
public class HamCycState
	extends Tuple
	{

// Hidden class data members.

	// The graph being searched.
	static AMGraph graph;
	static int V;

	// Search level threshold at which to switch from BFS to DFS.
	static int threshold;

	// Flag to stop search.
	static volatile boolean stop;

// Hidden instance data members.

	// Vertices in the path.
	private int[] path;

	// Search level = index of last vertex in the path.
	private int level;

// Exported class operations.

	/**
	 * Specify the graph to be analyzed using DFS only.
	 *
	 * @param  graph  Graph.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>graph</TT> is null.
	 */
	public static void setGraph
		(AMGraph graph)
		{
		HamCycState.graph = graph;
		HamCycState.V = graph.V();
		HamCycState.threshold = 0;
		HamCycState.stop = false;
		}

// Exported constructors.

	/**
	 * Construct a new search state object that supports DFS only. All instances
	 * of class HamCycState will analyze the graph specified in the {@link
	 * #setGraph(AMGraph) setGraph()} method.
	 */
	public HamCycState()
		{
		path = new int [V];
		for (int i = 0; i < V; ++ i)
			path[i] = i;
		level = 0;
		}

// Exported operations.

	/**
	 * Clone this search state object.
	 *
	 * @return  Clone.
	 */
	public Object clone()
		{
		HamCycState graph = (HamCycState) super.clone();
		graph.path = (int[]) this.path.clone();
		return graph;
		}

	/**
	 * Search the graph from this state.
	 *
	 * @return  Search state object containing the Hamiltonian cycle found, or
	 *          null if a Hamiltonian cycle was not found.
	 */
	public HamCycState search()
		{
		return (level < threshold) ? bfs() : dfs();
		}

	/**
	 * Stop the search in progress.
	 */
	public static void stop()
		{
		stop = true;
		}

	/**
	 * Returns a string version of this search state object.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		StringBuilder b = new StringBuilder();
		Formatter f = new Formatter (b);
		for (int i = 0; i <= level; ++ i)
			{
			if (i > 0) f.format (" ");
			f.format ("%d", path[i]);
			}
		return b.toString();
		}

	/**
	 * Write this search state object to the given out stream.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeOut
		(OutStream out)
		throws IOException
		{
		out.writeIntArray (path);
		out.writeInt (level);
		}

	/**
	 * Read this search state object from the given in stream.
	 *
	 * @param  in  In stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readIn
		(InStream in)
		throws IOException
		{
		path = in.readIntArray();
		level = in.readInt();
		if (path.length != V)
			throw new IOException (String.format
				("HamCycState.readIn(): Mismatch: V = %d, path.length = %d",
				 V, path.length));
		}

// Hidden operations.

	/**
	 * Do a breadth first search of the graph from this state.
	 *
	 * @return  Search state object containing the Hamiltonian cycle found, or
	 *          null if a Hamiltonian cycle was not found.
	 */
	private HamCycState bfs()
		{
		// Try extending the path to each vertex adjacent to the current
		// vertex.
		for (int i = level + 1; i < V && ! stop; ++ i)
			if (adjacent (i))
				{
				++ level;
				swap (level, i);
				enqueue ((HamCycState) this.clone());
				-- level;
				}
		return null;
		}

	/**
	 * Do a depth first search of the graph from this state.
	 *
	 * @return  Search state object containing the Hamiltonian cycle found, or
	 *          null if a Hamiltonian cycle was not found.
	 */
	private HamCycState dfs()
		{
		// Base case: Check if there is an edge from the last vertex to the
		// first vertex.
		if (level == V - 1)
			{
			if (adjacent (0))
				return this;
			}

		// Recursive case: Try extending the path to each vertex adjacent to
		// the current vertex.
		else
			{
			for (int i = level + 1; i < V && ! stop; ++ i)
				if (adjacent (i))
					{
					++ level;
					swap (level, i);
					if (dfs() != null)
						return this;
					-- level;
					}
			}

		return null;
		}

	/**
	 * Determine if the given path element is adjacent to the current path
	 * element.
	 *
	 * @return  True if adjacent, false if not.
	 */
	private boolean adjacent
		(int a)
		{
		return graph.isAdjacent (path[level], path[a]);
		}

	/**
	 * Swap the given path elements.
	 *
	 * @param  a  Index of first path element to swap.
	 * @param  b  Index of second path element to swap.
	 */
	private void swap
		(int a,
		 int b)
		{
		int t = path[a];
		path[a] = path[b];
		path[b] = t;
		}

	/**
	 * Enqueue the given search state object during a BFS.
	 * <P>
	 * The base class <TT>enqueue()</TT> method throws an unsupported operation
	 * exception, because the base class does not support BFS. A subclass may
	 * override the <TT>enqueue()</TT> method to place the search state object
	 * in a queue.
	 *
	 * @param  state  Search state object to enqueue.
	 */
	protected void enqueue
		(HamCycState state)
		{
		throw new UnsupportedOperationException();
		}

	}
