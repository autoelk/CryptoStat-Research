//******************************************************************************
//
// File:    GraphSpec.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.GraphSpec
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

package edu.rit.util;

import java.util.Iterator;

/**
 * Interface GraphSpec specifies the interface for an object that specifies a
 * graph. To obtain the specifications for a graph:
 * <OL TYPE=1>
 * <P><LI>
 * Call the {@link #V() V()} method to obtain the number of vertices.
 * <P><LI>
 * Call the {@link #E() E()} method to obtain the number of edges.
 * <P><LI>
 * Repeatedly call the {@link #hasNext() hasNext()} and {@link #next() next()}
 * methods to obtain the edges themselves.
 * </OL>
 *
 * @author  Alan Kaminsky
 * @version 27-Apr-2018
 */
public interface GraphSpec
	extends Iterator<GraphSpec.Edge>
	{

// Exported helper class.

	/**
	 * Class GraphSpec.Edge encapsulates one edge in a graph. The fields specify
	 * the two vertices linked by the edge. Each vertex is an integer in the
	 * range 0 through <I>V</I>&minus;1, where <I>V</I> is the number of
	 * vertices in the graph.
	 */
	public static class Edge
		{
		/**
		 * First vertex.
		 */
		public int v1;

		/**
		 * Second vertex.
		 */
		public int v2;

		/**
		 * Construct a new uninitialized edge.
		 */
		public Edge()
			{
			}

		/**
		 * Construct a new edge.
		 *
		 * @param  v1  First vertex.
		 * @param  v2  Second vertex.
		 */
		public Edge
			(int v1,
			 int v2)
			{
			this.v1 = v1;
			this.v2 = v2;
			}
		}

// Exported operations.

	/**
	 * Get the number of vertices in this graph specification.
	 *
	 * @return  Number of vertices.
	 */
	public int V();

	/**
	 * Get the number of edges in this graph specification.
	 *
	 * @return  Number of edges.
	 */
	public int E();

	/**
	 * Reset this graph specification. This restarts the iteration over the
	 * edges, such that calling <TT>hasNext()</TT> and <TT>next()</TT> will
	 * generate a fresh graph with the same number of vertices <I>V</I> and
	 * the same number of edges <I>E</I>.
	 */
	public void reset();

	/**
	 * Determine if there are more edges.
	 *
	 * @return  True if there are more edges, false if not.
	 */
	public boolean hasNext();

	/**
	 * Get the next edge.
	 * <P>
	 * <I>Note:</I> The <TT>next()</TT> method is permitted to return the
	 * <I>same Edge object</I>, with different vertices, on every call. Extract
	 * the vertices from the returned edge object and store them in another data
	 * structure; do not store a reference to the returned edge object itself.
	 *
	 * @return  Next edge.
	 *
	 * @exception  NoSuchElementException
	 *     (unchecked exception) Thrown if there are no more edges.
	 */
	public GraphSpec.Edge next();

	}
