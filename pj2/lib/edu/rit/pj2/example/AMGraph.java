//******************************************************************************
//
// File:    AMGraph.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.AMGraph
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

import edu.rit.util.BitSet;
import edu.rit.util.GraphSpec;

/**
 * Class AMGraph provides an adjacency matrix data structure for a graph with
 * any number of vertices.
 *
 * @author  Alan Kaminsky
 * @version 27-Apr-2018
 */
public class AMGraph
	{

// Hidden data members.

	// Number of vertices.
	private int V;

	// The graph's adjacency matrix. adjacent[i] is the set of vertices adjacent
	// to vertex i.
	private BitSet[] adjacent;

// Exported constructors.

	/**
	 * Construct a new uninitialized adjacency matrix.
	 */
	public AMGraph()
		{
		}

	/**
	 * Construct a new adjacency matrix for the given graph specification.
	 *
	 * @param  gspec  Graph specification.
	 */
	public AMGraph
		(GraphSpec gspec)
		{
		set (gspec);
		}

// Exported operations.

	/**
	 * Set this adjacency matrix to the given graph specification.
	 *
	 * @param  gspec  Graph specification.
	 */
	public void set
		(GraphSpec gspec)
		{
		V = gspec.V();
		if (V < 0)
			throw new IllegalArgumentException (String.format
				("AMGraph.set(): V = %d illegal", V));
		if (adjacent == null || adjacent.length != V)
			{
			adjacent = new BitSet [V];
			for (int i = 0; i < V; ++ i)
				adjacent[i] = new BitSet (V);
			}
		else
			{
			for (int i = 0; i < V; ++ i)
				adjacent[i].clear();
			}
		while (gspec.hasNext())
			{
			GraphSpec.Edge edge = gspec.next();
			adjacent[edge.v1].add (edge.v2);
			adjacent[edge.v2].add (edge.v1);
			}
		}

	/**
	 * Get the number of vertices in this graph.
	 *
	 * @return  Number of vertices.
	 */
	public int V()
		{
		return V;
		}

	/**
	 * Determine if the given vertices are adjacent in this graph.
	 *
	 * @param  v1  First vertex.
	 * @param  v2  Second vertex.
	 *
	 * @return  True if <TT>v1</TT> is adjacent to <TT>v2</TT>, false otherwise.
	 */
	public boolean isAdjacent
		(int v1,
		 int v2)
		{
		return adjacent[v1].contains (v2);
		}

	/**
	 * Determine if the given vertex set is a vertex cover for this graph.
	 *
	 * @param  vset  Vertex set.
	 */
	public boolean isVertexCover
		(BitSet vset)
		{
		boolean covered = true;
		for (int i = 0; covered && i < V; ++ i)
			if (! vset.contains (i))
				covered = adjacent[i].isSubsetOf (vset);
		return covered;
		}

	}
