//******************************************************************************
//
// File:    RandomGraph.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.RandomGraph
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

import java.util.NoSuchElementException;

/**
 * Class RandomGraph is a {@linkplain GraphSpec GraphSpec} object that specifies
 * a random graph. Given a number of vertices <I>V</I> and a number of edges
 * <I>E,</I> a RandomGraph object specifies a graph consisting of a randomly
 * chosen subset of size <I>E</I> of all possible edges among <I>V</I> vertices.
 *
 * @author  Alan Kaminsky
 * @version 27-Apr-2018
 */
public class RandomGraph
	implements GraphSpec
	{

// Hidden data members.

	private int V;
	private int E;
	private Random prng;

	private GraphSpec.Edge edge;
	private int v1;
	private int v2;
	private int needed;
	private int available;

// Exported constructors.

	/**
	 * Construct a random graph specification with the given seed.
	 *
	 * @param  V     Number of vertices.
	 * @param  E     Number of edges.
	 * @param  seed  Random seed.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <I>V</I> &lt; 0. Thrown if <I>E</I>
	 *     &lt; 0. Thrown if <I>E</I> &gt; <I>V</I>(<I>V</I>&minus;1)/2.
	 */
	public RandomGraph
		(int V,
		 int E,
		 long seed)
		{
		this (V, E, new Random (seed));
		}

	/**
	 * Construct a random graph specification with the given pseudorandom number
	 * generator.
	 *
	 * @param  V     Number of vertices.
	 * @param  E     Number of edges.
	 * @param  prng  Pseudorandom number generator.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <I>V</I> &lt; 0. Thrown if <I>E</I>
	 *     &lt; 0. Thrown if <I>E</I> &gt; <I>V</I>(<I>V</I>&minus;1)/2.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>prng</TT> is null.
	 */
	public RandomGraph
		(int V,
		 int E,
		 Random prng)
		{
		if (V < 0)
			throw new IllegalArgumentException (String.format
				("RandomGraph(): V = %d illegal", V));
		if (E < 0)
			throw new IllegalArgumentException (String.format
				("RandomGraph(): E = %d illegal", E));
		if (E > maxE (V))
			throw new IllegalArgumentException (String.format
				("RandomGraph(): V = %d, E = %d illegal", V, E));
		if (prng == null)
			throw new NullPointerException
				("RandomGraph(): prng is null");

		this.V = V;
		this.E = E;
		this.prng = prng;
		this.edge = new GraphSpec.Edge();
		reset();
		}

// Exported operations.

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
	 * Get the number of edges in this graph.
	 *
	 * @return  Number of edges.
	 */
	public int E()
		{
		return E;
		}

	/**
	 * Reset this graph specification. This restarts the iteration over the
	 * edges, such that calling <TT>hasNext()</TT> and <TT>next()</TT> will
	 * generate a fresh graph with the same number of vertices <I>V</I> and
	 * the same number of edges <I>E</I>.
	 * <P>
	 * The <TT>reset()</TT> method in class RandomGraph will generate a fresh
	 * graph with different randomly chosen edges.
	 */
	public void reset()
		{
		v1 = -1;
		v2 = V - 1;
		needed = E;
		available = maxE (V);
		}

	/**
	 * Determine if there are more edges.
	 *
	 * @return  True if there are more edges, false if not.
	 */
	public boolean hasNext()
		{
		return needed > 0;
		}

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
	public GraphSpec.Edge next()
		{
		if (! hasNext())
			throw new NoSuchElementException
				("RandomGraph.next(): No more edges");
		for (;;)
			{
			++ v2;
			if (v2 == V)
				{
				++ v1;
				v2 = v1 + 1;
				}
			if (prng.nextDouble() < (double)needed/(double)available)
				{
				-- needed;
				-- available;
				break;
				}
			else
				{
				-- available;
				}
			}
		edge.v1 = v1;
		edge.v2 = v2;
		return edge;
		}

	/**
	 * Unsupported operation.
	 *
	 * @exception  UnsupportedOperationException
	 *     (unchecked exception) Thrown always.
	 */
	public void remove()
		{
		throw new UnsupportedOperationException
			("RandomGraph.remove(): Unsupported operation");
		}

// Hidden operations.

	/**
	 * Returns the maximum number of edges for a graph with <TT>V</TT> vertices.
	 */
	private static int maxE
		(int V)
		{
		return Math.max (0, V*(V - 1)/2);
		}

	}
