//******************************************************************************
//
// File:    Stack.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.Stack
//
// This Java source file is copyright (C) 2019 by Alan Kaminsky. All rights
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
 * Class Stack provides a first-in-first-out stack of items stored in a
 * dynamically-sized array.
 * <P>
 * Operations take constant time unless otherwise specified. <I>n</I> is the
 * number of items stored in the stack.
 *
 * @param  <T>  Stack item data type.
 *
 * @author  Alan Kaminsky
 * @version 05-Jun-2019
 */
public class Stack<T>
	implements Streamable
	{

// Hidden data members.

	// Chunk size for growing array.
	private static final int CHUNK = 8;

	// Array of stack items and maximum size.
	T[] item;
	int max;

	// Number of stack items. Top index = size - 1.
	int size;

// Exported constructors.

	/**
	 * Construct a new stack.
	 */
	public Stack()
		{
		item = allocate (CHUNK);
		max = item.length;
		}

	/**
	 * Construct a new stack that is a shallow copy of the given stack. The new
	 * stack's entries contain the same items in the same order as the given
	 * stack. Time: <I>O</I>(<I>n</I>).
	 *
	 * @param  stack  Stack to copy.
	 */
	public Stack
		(Stack<T> stack)
		{
		copy (stack);
		}

// Exported operations.

	/**
	 * Set this stack to a shallow copy of the given stack. Afterwards, this
	 * stack contains the same items in the same order as the given stack. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  stack  Stack to copy.
	 */
	public void copy
		(Stack<T> stack)
		{
		if (this.max < stack.size)
			{
			this.item = allocate (stack.size);
			this.max = this.item.length;
			}
		System.arraycopy (stack.item, 0, this.item, 0, stack.size);
		this.size = stack.size;
		}

	/**
	 * Determine if this stack is empty.
	 *
	 * @return  True if this stack is empty, false if it isn't.
	 */
	public boolean isEmpty()
		{
		return size == 0;
		}

	/**
	 * Returns the number of items in this stack.
	 *
	 * @return  Number of items.
	 */
	public int size()
		{
		return size;
		}

	/**
	 * Clear this stack.
	 */
	public void clear()
		{
		size = 0;
		}

	/**
	 * Push the given item onto the top of this stack.
	 *
	 * @param  i  Item to push.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this stack is full.
	 */
	public void push
		(T i)
		{
		if (size == max)
			{
			T[] newitem = allocate (size + 1);
			System.arraycopy (item, 0, newitem, 0, size);
			item = newitem;
			max = newitem.length;
			}
		item[size] = i;
		++ size;
		}

	/**
	 * Pop the item off the top of this stack.
	 *
	 * @return  Item that was popped.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this stack is empty.
	 */
	public T pop()
		{
		if (isEmpty())
			throw new IllegalStateException
				("Stack.pop(): Stack is empty");
		-- size;
		return item[size];
		}

	/**
	 * Get the item at the top of this stack.
	 *
	 * @return  Top item.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this stack is empty.
	 */
	public T top()
		{
		if (isEmpty())
			throw new IllegalStateException
				("Stack.top(): Stack is empty");
		return item[size-1];
		}

	/**
	 * Perform the given action on each item in this stack. For each item in
	 * this stack from top to bottom, the given <TT>action</TT>'s <TT>run()</TT>
	 * method is called, passing in the item.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> pushes or pops items, the
	 * <TT>forEachItemDo()</TT> method's behavior is not specified.
	 *
	 * @param  action  Action.
	 */
	public void forEachItemDo
		(Action<T> action)
		{
		for (int i = size - 1; i >= 0; -- i)
			action.run (item[i]);
		}

	/**
	 * Perform the given action on each item in this stack and return a result.
	 * For each item in this stack from top to bottom, the given
	 * <TT>action</TT>'s <TT>run()</TT> method is called, passing in the item.
	 * After all the stack items have been processed, the given
	 * <TT>action</TT>'s <TT>result()</TT> method is called, and its result is
	 * returned.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> pushes or pops items, the
	 * <TT>forEachItemDo()</TT> method's behavior is not specified.
	 *
	 * @param  <R>     Result data type.
	 * @param  action  Action.
	 *
	 * @return  Result of processing all the stack items.
	 */
	public <R> R forEachItemDo
		(ActionResult<T,R> action)
		{
		for (int i = size - 1; i >= 0; -- i)
			action.run (item[i]);
		return action.result();
		}

	/**
	 * Store this stack's items in the given array. The top item is stored at
	 * index 0, the next item at index 1, and so on. The number of array
	 * elements set is <TT>array.length</TT>. If this stack contains fewer than
	 * <TT>array.length</TT> items, the remaining array elements are set to 0.
	 *
	 * @param  array  Array in which to store items.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 */
	public T[] toArray
		(T[] array)
		{
		return toArray (array, 0, array.length);
		}

	/**
	 * Store this stack's items in the given array. The top item is stored at
	 * index <TT>off</TT>, the next item at index <TT>off</TT>+1, and so on. The
	 * number of array elements set is <TT>len</TT>. If this stack contains
	 * fewer than <TT>len</TT> items, the remaining array elements are set to 0.
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
	public T[] toArray
		(T[] array,
		 int off,
		 int len)
		{
		if (off < 0 || len < 0 || off+len > array.length)
			throw new IndexOutOfBoundsException();
		int n = Math.min (size, len);
		for (int i = 0; i < n; ++ i)
			array[off+i] = item[size-1-i];
		if (n < len) Arrays.fill (array, off + n, off + len, 0);
		return array;
		}

	/**
	 * Write this object's fields to the given out stream. The stack elements
	 * are written using {@link OutStream#writeReference(Object)
	 * writeReference()}.
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
		for (int i = 0; i < size; ++ i)
			out.writeReference (item[i]);
		}

	/**
	 * Read this object's fields from the given in stream. The stack elements
	 * are read using {@link InStream#readReference() readReference()}.
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
		for (int i = 0; i < size; ++ i)
			item[i] = (T) in.readReference();
		}

// Hidden operations.

	/**
	 * Allocate an array big enough to store the given stack size.
	 *
	 * @param  size  Queue size.
	 *
	 * @return  Newly allocated array.
	 */
	private T[] allocate
		(int size)
		{
		int len = size + CHUNK - 1;
		len = Math.max (len/CHUNK, 1);
		len *= CHUNK;
		return (T[]) new Object [len];
		}

	}
