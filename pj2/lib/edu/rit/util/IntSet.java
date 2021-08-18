//******************************************************************************
//
// File:    IntSet.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.IntSet
//
// This Java source file is copyright (C) 2017 by Alan Kaminsky. All rights
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

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import edu.rit.util.IntAction;
import edu.rit.util.IntActionResult;
import edu.rit.util.Random;
import java.io.IOException;
import java.util.Arrays;

/**
 * Class IntSet provides a set of integers. The set can contain integers in the
 * range 0 to <I>N</I>&minus;1. In addition to the usual set operations, class
 * IntSet provides efficient operations to add or remove randomly chosen
 * elements and to select randomly chosen elements or non-elements.
 * <P>
 * In the method descriptions below, "elements" refers to the integers in the
 * range 0 to <I>N</I>&minus;1 that are contained in the set. "Non-elements"
 * refers to the integers in the range 0 to <I>N</I>&minus;1 that are not
 * contained in the set. An "empty" set is one that contains no integers. A
 * "saturated" set is one that contains all the integers 0 to <I>N</I>&minus;1.
 *
 * @author  Alan Kaminsky
 * @version 30-Aug-2017
 */
public class IntSet
	implements Streamable
	{

// Hidden data members.

	// Maximum number of elements in this set.
	private int N;

	// Number of elements in this set.
	private int size;

	// Array of elements in this set. elem[0] .. elem[size-1] contain the
	// elements in no particular order. elem[size] .. elem[N-1] contain the
	// non-elements in no particular order.
	private int[] elem;

	// Array of indexes of elements. For all i, elem[index[i]] = i.
	private int[] index;

// Exported constructors.

	/**
	 * Construct a new empty set. The set can hold integers in the range 0 to
	 * <I>N</I>&minus;1.
	 *
	 * @param  N  Maximum number of elements.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <I>N</I> &le; 0.
	 */
	public IntSet
		(int N)
		{
		if (N <= 0)
			throw new IllegalArgumentException (String.format
				("IntSet(): N = %d illegal", N));
		initialize (N);
		}

	private void initialize
		(int N)
		{
		this.N = N;
		this.size = 0;
		this.elem = new int [N];
		this.index = new int [N];
		initializeElem();
		}

	private void initializeElem()
		{
		for (int i = 0; i < N; ++ i)
			elem[i] = index[i] = i;
		}

	/**
	 * Construct a new set that is a copy of the given set.
	 *
	 * @param  set  Set to copy.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>set</TT> is null.
	 */
	public IntSet
		(IntSet set)
		{
		copyUnequalSize (set);
		}

// Exported operations.

	/**
	 * Clear this set. Afterwards, this set contains no elements.
	 *
	 * @return  This set.
	 */
	public IntSet clear()
		{
		this.size = 0;
		initializeElem();
		return this;
		}

	/**
	 * Saturate this set. Afterwards, this set contains elements 0 to
	 * <I>N</I>&minus;1.
	 *
	 * @return  This set.
	 */
	public IntSet saturate()
		{
		this.size = N;
		initializeElem();
		return this;
		}

	/**
	 * Make this set be a copy of the given set.
	 *
	 * @param  set  Set to copy.
	 *
	 * @return  This set.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>set</TT> is null.
	 */
	public IntSet copy
		(IntSet set)
		{
		return (this.N == set.N) ? copyEqualSize (set) : copyUnequalSize (set);
		}

	private IntSet copyEqualSize
		(IntSet set)
		{
		this.size = set.size;
		System.arraycopy (set.elem, 0, this.elem, 0, N);
		System.arraycopy (set.index, 0, this.index, 0, N);
		return this;
		}

	private IntSet copyUnequalSize
		(IntSet set)
		{
		this.N = set.N;
		this.size = set.size;
		this.elem = (int[]) set.elem.clone();
		this.index = (int[]) set.index.clone();
		return this;
		}

	/**
	 * Determine if this set is empty.
	 *
	 * @return  True if this set is empty, false otherwise.
	 */
	private boolean isEmpty()
		{
		return size == 0;
		}

	/**
	 * Determine if this set is saturated.
	 *
	 * @return  True if this set is saturated, false otherwise.
	 */
	private boolean isSaturated()
		{
		return size == N;
		}

	/**
	 * Returns the number of elements in this set.
	 *
	 * @return  Number of elements.
	 */
	public int size()
		{
		return size;
		}

	/**
	 * Determine if this set contains the given element.
	 *
	 * @param  e  Element.
	 *
	 * @return  True if this set contains <I>e</I>, false otherwise.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <I>e</I> is not in the range 0 to
	 *     <I>N</I>&minus;1.
	 */
	public boolean contains
		(int e)
		{
		return index[e] < size;
		}

	/**
	 * Add the given element to this set.
	 *
	 * @param  e  Element.
	 *
	 * @return  This set.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <I>e</I> is not in the range 0 to
	 *     <I>N</I>&minus;1.
	 */
	public IntSet add
		(int e)
		{
		if (! contains (e))
			addNonElement (e);
		return this;
		}

	/**
	 * Add to this set a randomly-chosen element that is not presently in this
	 * set. The return value is the element that was added. If this set is
	 * saturated, this set is not altered and &minus;1 is returned.
	 *
	 * @param  prng  Pseudorandom number generator.
	 *
	 * @return  Element that was added, or &minus;1 if this set is saturated.
	 */
	public int add
		(Random prng)
		{
		int e = randomNonElement (prng);
		if (e != -1)
			addNonElement (e);
		return e;
		}

	private void addNonElement
		(int e)
		{
		int i = index[e];
		int ip = size;
		int ep = elem[ip];
		elem[i] = ep;
		elem[ip] = e;
		index[e] = ip;
		index[ep] = i;
		++ size;
		}

	/**
	 * Remove the given element from this set.
	 *
	 * @param  e  Element.
	 *
	 * @return  This set.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <I>e</I> is not in the range 0 to
	 *     <I>N</I>&minus;1.
	 */
	public IntSet remove
		(int e)
		{
		if (contains (e))
			removeElement (e);
		return this;
		}

	/**
	 * Remove from this set a randomly-chosen element that is presently in this
	 * set. The return value is the element that was removed. If this set is
	 * empty, this set is not altered and &minus;1 is returned.
	 *
	 * @param  prng  Pseudorandom number generator.
	 *
	 * @return  Element that was removed, or -1 if this set is empty.
	 */
	public int remove
		(Random prng)
		{
		int e = randomElement (prng);
		if (e != -1)
			removeElement (e);
		return e;
		}

	private void removeElement
		(int e)
		{
		int i = index[e];
		int ip = size - 1;
		int ep = elem[ip];
		elem[i] = ep;
		elem[ip] = e;
		index[e] = ip;
		index[ep] = i;
		-- size;
		}

	/**
	 * Returns a randomly chosen element in this set.
	 *
	 * @param  prng  Pseudorandom number generator.
	 *
	 * @return  Element, or &minus;1 if this set is empty.
	 */
	public int randomElement
		(Random prng)
		{
		return (isEmpty()) ? -1 : elem [prng.nextInt (size)];
		}

	/**
	 * Returns a randomly chosen element not in this set.
	 *
	 * @param  prng  Pseudorandom number generator.
	 *
	 * @return  Non-element, or &minus;1 if this set is saturated.
	 */
	public int randomNonElement
		(Random prng)
		{
		return (isSaturated()) ? -1 : elem [size + prng.nextInt (N - size)];
		}

	/**
	 * Iterate over the elements in this set. For each element <I>e</I> in this
	 * set, the given <TT>action</TT>'s <TT>run</TT> method is called with
	 * <I>e</I> as the argument. The order in which elements are processed is
	 * not specified.
	 *
	 * @param  action  Action.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>action</TT> is null.
	 */
	public void forEachElementDo
		(IntAction action)
		{
		for (int i = 0; i < size; ++ i)
			action.run (elem[i]);
		}

	/**
	 * Iterate over the elements in this set and return a result. For each
	 * element <I>e</I> in this set, the given <TT>action</TT>'s <TT>run</TT>
	 * method is called with <I>e</I> as the argument. The order in which
	 * elements are processed is not specified. Then the value returned by the
	 * given <TT>action</TT>'s <TT>result()</TT> method is returned.
	 *
	 * @param  <R>     Result data type.
	 * @param  action  Action.
	 *
	 * @return  Action result.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>action</TT> is null.
	 */
	public <R> R forEachElementDo
		(IntActionResult<R> action)
		{
		for (int i = 0; i < size; ++ i)
			action.run (elem[i]);
		return action.result();
		}

	/**
	 * Iterate over the non-elements in this set. For each element <I>e</I> not
	 * in this set, the given <TT>action</TT>'s <TT>run</TT> method is called
	 * with <I>e</I> as the argument. The order in which non-elements are
	 * processed is not specified.
	 *
	 * @param  action  Action.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>action</TT> is null.
	 */
	public void forEachNonElementDo
		(IntAction action)
		{
		for (int i = size; i < N; ++ i)
			action.run (elem[i]);
		}

	/**
	 * Iterate over the non-elements in this set and return a result. For each
	 * element <I>e</I> not in this set, the given <TT>action</TT>'s
	 * <TT>run</TT> method is called with <I>e</I> as the argument. The order in
	 * which non-elements are processed is not specified. Then the value
	 * returned by the given <TT>action</TT>'s <TT>result()</TT> method is
	 * returned.
	 *
	 * @param  <R>     Result data type.
	 * @param  action  Action.
	 *
	 * @return  Action result.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>action</TT> is null.
	 */
	public <R> R forEachNonElementDo
		(IntActionResult<R> action)
		{
		for (int i = size; i < N; ++ i)
			action.run (elem[i]);
		return action.result();
		}

	/**
	 * Get the given element in this set. Elements are indexed from 0 to {@link
	 * #size()}&minus;1. The order of the elements is not specified.
	 *
	 * @param  i  Index (0 to {@link #size()}&minus;1).
	 *
	 * @return  Element at index <TT>i</TT>.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>i</TT> is outside the above
	 *     range.
	 */
	public int element
		(int i)
		{
		if (i < 0 || i >= size)
			throw new IndexOutOfBoundsException (String.format
				("IntSet.element(): I = %d out of bounds", i));
		return elem[i];
		}

	/**
	 * Get the given non-element in this set. Non-elements are indexed from 0 to
	 * <I>N</I>&minus;{@link #size()}&minus;1. The order of the non-elements is
	 * not specified.
	 *
	 * @param  i  Index (0 to <I>N</I>&minus;{@link #size()}&minus;1).
	 *
	 * @return  Non-element at index <TT>i</TT>.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>i</TT> is outside the above
	 *     range.
	 */
	public int nonElement
		(int i)
		{
		if (i < 0 || i >= N - size)
			throw new IndexOutOfBoundsException (String.format
				("IntSet.nonElement(): I = %d out of bounds", i));
		return elem[size+i];
		}

	/**
	 * Store this set's elements in the given array. This set's elements are
	 * stored starting at array index 0. The order in which this set's elements
	 * appear in the array is not specified. The number of array elements stored
	 * is <TT>array.length</TT>. If this set contains fewer than
	 * <TT>array.length</TT> elements, the remaining array elements are set to
	 * 0.
	 *
	 * @param  array  Array in which to store elements.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 */
	public int[] toArray
		(int[] array)
		{
		return toArray (array, 0, array.length);
		}

	/**
	 * Store this set's elements in a portion of the given array. This set's
	 * elements are stored starting at array index <TT>off</TT>. The order in
	 * which this set's elements appear in the array is not specified. The
	 * number of array elements stored is <TT>len</TT>. If this set contains
	 * fewer than <TT>len</TT> elements, the remaining array elements are set to
	 * 0.
	 *
	 * @param  array  Array in which to store elements.
	 * @param  off    Index at which to store first element.
	 * @param  len    Number of elements to store.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>array.length</TT>.
	 */
	public int[] toArray
		(int[] array,
		 int off,
		 int len)
		{
		if (off < 0 || len < 0 || off+len > array.length)
			{
			throw new IndexOutOfBoundsException();
			}
		int n = Math.min (size, len);
		System.arraycopy (elem, 0, array, off, n);
		if (n < len) Arrays.fill (array, off + n, off + len, 0);
		return array;
		}

	/**
	 * Store this set's non-elements in the given array. This set's non-elements
	 * are stored starting at array index 0. The order in which this set's
	 * non-elements appear in the array is not specified. The number of array
	 * elements stored is <TT>array.length</TT>. If this set has fewer than
	 * <TT>array.length</TT> non-elements, the remaining array elements are set
	 * to 0.
	 *
	 * @param  array  Array in which to store non-elements.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 */
	public int[] nonElementsToArray
		(int[] array)
		{
		return nonElementsToArray (array, 0, array.length);
		}

	/**
	 * Store this set's non-elements in a portion of the given array. This set's
	 * non-elements are stored starting at array index <TT>off</TT>. The order
	 * in which this set's non-elements appear in the array is not specified.
	 * The number of array elements stored is <TT>len</TT>. If this set has
	 * fewer than <TT>len</TT> non-elements, the remaining array elements are
	 * set to 0.
	 *
	 * @param  array  Array in which to store non-elements.
	 * @param  off    Index at which to store first non-element.
	 * @param  len    Number of non-elements to store.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>array.length</TT>.
	 */
	public int[] nonElementsToArray
		(int[] array,
		 int off,
		 int len)
		{
		if (off < 0 || len < 0 || off+len > array.length)
			{
			throw new IndexOutOfBoundsException();
			}
		int n = Math.min (N - size, len);
		System.arraycopy (elem, size, array, off, n);
		if (n < len) Arrays.fill (array, off + n, off + len, 0);
		return array;
		}

	/**
	 * Write this set to the given out stream.
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
		out.writeInt (N);
		out.writeInt (size);
		for (int i = 0; i < size; ++ i)
			out.writeInt (elem[i]);
		}

	/**
	 * Read this set from the given in stream.
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
		initialize (in.readInt());
		int sz = in.readInt();
		for (int i = 0; i < sz; ++ i)
			add (in.readInt());
		}

	}
