//******************************************************************************
//
// File:    LongQueue.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.LongQueue
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

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import java.io.IOException;
import java.util.Arrays;

/**
 * Class LongQueue provides a first-in-first-out queue of long integers (type
 * <TT>long</TT>) stored in a dynamically-sized array.
 * <P>
 * Operations take constant time unless otherwise specified. <I>n</I> is the
 * number of items stored in the queue.
 *
 * @author  Alan Kaminsky
 * @version 07-Nov-2018
 */
public class LongQueue
	implements Streamable
	{

// Hidden data members.

	// Chunk size for growing array.
	private static final int CHUNK = 8;

	// Array of queue items and maximum size.
	long[] item;
	int max;

	// Number of queue items, read index, write index.
	int size;
	int rd;
	int wr;

// Exported constructors.

	/**
	 * Construct a new queue.
	 */
	public LongQueue()
		{
		item = allocate (CHUNK);
		max = item.length;
		}

	/**
	 * Construct a new queue that is a copy of the given queue. The new queue's
	 * entries contain the same items in the same order as the given queue.
	 * Time: <I>O</I>(<I>n</I>).
	 *
	 * @param  queue  Queue to copy.
	 */
	public LongQueue
		(LongQueue queue)
		{
		copy (queue);
		}

// Exported operations.

	/**
	 * Set this queue to a copy of the given queue. Afterwards, this queue
	 * contains the same items in the same order as the given queue. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  queue  Queue to copy.
	 */
	public void copy
		(LongQueue queue)
		{
		if (this.max < queue.size)
			{
			this.item = allocate (queue.size);
			this.max = this.item.length;
			}
		int from = queue.rd;
		int to = 0;
		for (int i = 0; i < queue.size; ++ i)
			{
			this.item[to] = queue.item[from];
			from = incr (from, queue.max);
			to = incr (to, this.max);
			}
		this.size = queue.size;
		this.rd = 0;
		this.wr = to;
		}

	/**
	 * Determine if this queue is empty.
	 *
	 * @return  True if this queue is empty, false if it isn't.
	 */
	public boolean isEmpty()
		{
		return size == 0;
		}

	/**
	 * Returns the number of items in this queue.
	 *
	 * @return  Number of items.
	 */
	public int size()
		{
		return size;
		}

	/**
	 * Clear this queue.
	 */
	public void clear()
		{
		size = 0;
		rd = 0;
		wr = 0;
		}

	/**
	 * Push the given item onto the back of this queue.
	 *
	 * @param  i  Item to push.
	 */
	public void push
		(long i)
		{
		if (size == max)
			{
			long[] newitem = allocate (size + 1);
			int newmax = newitem.length;
			int from = rd;
			int to = 0;
			for (int j = 0; j < size; ++ j)
				{
				newitem[to] = item[from];
				from = incr (from, max);
				to = incr (to, newmax);
				}
			item = newitem;
			max = newmax;
			rd = 0;
			wr = to;
			}
		item[wr] = i;
		++ size;
		wr = incr (wr, max);
		}

	/**
	 * Pop the first item off the front of this queue.
	 *
	 * @return  Item that was popped.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this queue is empty.
	 */
	public long pop()
		{
		if (isEmpty())
			throw new IllegalStateException
				("LongQueue.pop(): Queue is empty");
		long i = item[rd];
		-- size;
		rd = incr (rd, max);
		return i;
		}

	/**
	 * Get the item at the front of this queue.
	 *
	 * @return  Front item.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this queue is empty.
	 */
	public long front()
		{
		if (isEmpty())
			throw new IllegalStateException
				("LongQueue.front(): Queue is empty");
		return item[rd];
		}

	/**
	 * Perform the given action on each item in this queue. For each item in
	 * this queue from front to back, the given <TT>action</TT>'s <TT>run()</TT>
	 * method is called, passing in the item.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> pushes or pops items, the
	 * <TT>forEachItemDo()</TT> method's behavior is not specified.
	 *
	 * @param  action  Action.
	 */
	public void forEachItemDo
		(LongAction action)
		{
		int from = rd;
		for (int i = 0; i < size; ++ i)
			{
			action.run (item[from]);
			from = incr (from, max);
			}
		}

	/**
	 * Perform the given action on each item in this queue and return a result.
	 * For each item in this queue from front to back, the given
	 * <TT>action</TT>'s <TT>run()</TT> method is called, passing in the item.
	 * After all the queue items have been processed, the given
	 * <TT>action</TT>'s <TT>result()</TT> method is called, and its result is
	 * returned.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> pushes or pops items, the
	 * <TT>forEachItemDo()</TT> method's behavior is not specified.
	 *
	 * @param  <R>     Result data type.
	 * @param  action  Action.
	 *
	 * @return  Result of processing all the queue items.
	 */
	public <R> R forEachItemDo
		(LongActionResult<R> action)
		{
		int from = rd;
		for (int i = 0; i < size; ++ i)
			{
			action.run (item[from]);
			from = incr (from, max);
			}
		return action.result();
		}

	/**
	 * Store this queue's items in the given array. The first (front) item is
	 * stored at index 0, the second item at index 1, and so on. The number of
	 * array elements set is <TT>array.length</TT>. If this queue contains fewer
	 * than <TT>array.length</TT> items, the remaining array elements are set to
	 * 0.
	 *
	 * @param  array  Array in which to store items.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 */
	public long[] toArray
		(long[] array)
		{
		return toArray (array, 0, array.length);
		}

	/**
	 * Store this queue's items in the given array. The first (front) item is
	 * stored at index <TT>off</TT>, the second item at index <TT>off</TT>+1,
	 * and so on. The number of array elements set is <TT>len</TT>. If this
	 * queue contains fewer than <TT>len</TT> items, the remaining array
	 * elements are set to 0.
	 *
	 * @param  array  Array in which to store items.
	 * @param  off    Index at which to store first item.
	 * @param  len    Number of items to store.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>array.length</TT>.
	 */
	public long[] toArray
		(long[] array,
		 int off,
		 int len)
		{
		if (off < 0 || len < 0 || off+len > array.length)
			throw new IndexOutOfBoundsException();
		int n = Math.min (size, len);
		int from = rd;
		for (int i = 0; i < n; ++ i)
			{
			array[off+i] = item[from];
			from = incr (from, max);
			}
		if (n < len) Arrays.fill (array, off + n, off + len, 0);
		return array;
		}

	/**
	 * Write this object's fields to the given out stream.
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
		out.writeInt (size);
		int from = rd;
		for (int i = 0; i < size; ++ i)
			{
			out.writeLong (item[from]);
			from = incr (from, max);
			}
		}

	/**
	 * Read this object's fields from the given in stream.
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
		size = in.readInt();
		if (max < size)
			{
			item = allocate (size);
			max = item.length;
			}
		rd = 0;
		wr = 0;
		for (int i = 0; i < size; ++ i)
			{
			item[wr] = in.readLong();
			wr = incr (wr, max);
			}
		}

// Hidden operations.

	/**
	 * Allocate an array big enough to store the given queue size.
	 *
	 * @param  size  Queue size.
	 *
	 * @return  Newly allocated array.
	 */
	private long[] allocate
		(int size)
		{
		int len = size + CHUNK - 1;
		len = Math.max (len/CHUNK, 1);
		len *= CHUNK;
		return new long [len];
		}

	/**
	 * Returns (i + 1) mod m.
	 */
	private int incr
		(int i,
		 int m)
		{
		++ i;
		if (i == m) i = 0;
		return i;
		}

	}
