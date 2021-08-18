//******************************************************************************
//
// File:    SubsetSum.java
// Package: edu.rit.gpu.example
// Unit:    Class edu.rit.gpu.example.SubsetSum
//
// This Java source file is copyright (C) 2016 by Alan Kaminsky. All rights
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

package edu.rit.gpu.example;

import edu.rit.util.AList;
import edu.rit.util.Random;
import java.util.NoSuchElementException;

/**
 * Class SubsetSum provides an object that defines a subset sum problem as a
 * knapsack problem. The definition consists of the knapsack capacity, the
 * number of items, and the weight and value of each item. There are <I>N</I>
 * Items. Each item's weight is chosen uniformly at random in the range 1
 * through <I>maxW</I> inclusive. Each item's value is equal to its weight. The
 * knapsack capacity is equal to the sum of the weights of the first <I>S</I>
 * items. The solution to the subset sum problem consists of the first <I>S</I>
 * items.
 *
 * @author  Alan Kaminsky
 * @version 08-Jun-2016
 */
public class SubsetSum
	implements KnapsackProblem
	{

// Hidden data members.

	private int N;
	private AList<WV> wv;
	private long C;
	private int index;

// Exported constructors.

	/**
	 * Construct a new strongly correlated knapsack problem.
	 *
	 * @param  N       Number of items, <I>N</I> &ge; 1.
	 * @param  S       Number of items for subset sum, 1 &le; <I>S</I> &le;
	 *                 <I>N</I>.
	 * @param  maxW    Maximum item weight, <I>maxW</I> &ge; 1.
	 * @param  seed    Pseudorandom number generator seed.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if any argument is not in the stated
	 *     range.
	 */
	public SubsetSum
		(int N,
		 int S,
		 long maxW,
		 long seed)
		{
		if (N < 1)
			throw new IllegalArgumentException (String.format
				("SubsetSum(): N = %d illegal", N));
		if (1 > S || S > N)
			throw new IllegalArgumentException (String.format
				("SubsetSum(): S = %d illegal", S));
		if (maxW < 1)
			throw new IllegalArgumentException (String.format
				("KnapsackSC(): maxW = %d illegal", maxW));

		this.N = N;
		wv = new AList<WV>();
		C = 0L;
		Random prng = new Random (seed);
		for (int i = 0; i < N; ++ i)
			{
			long W = (long)(maxW*prng.nextDouble() + 1);
			wv.addLast (new WV (W, W));
			if (i < S) C += W;
			}
		index = 0;
		}

// Exported operations.

	/**
	 * Get the knapsack capacity.
	 *
	 * @return  Capacity.
	 */
	public long capacity()
		{
		return C;
		}

	/**
	 * Get the number of items, <I>N.</I>
	 *
	 * @return  Number of items.
	 */
	public int itemCount()
		{
		return N;
		}

	/**
	 * Get the weight and value of the next item. This method must be called
	 * <I>N</I> times to get all the items. Each method call returns a new
	 * {@linkplain WV} object.
	 *
	 * @return  Weight/value.
	 *
	 * @exception  NoSuchElementException
	 *     (unchecked exception) Thrown if this method is called more than
	 *     <I>N</I> times.
	 */
	public WV next()
		{
		if (index >= N)
			throw new NoSuchElementException
				("SubsetSum.next(): Called too many times");
		return wv.get (index ++);
		}

	}
