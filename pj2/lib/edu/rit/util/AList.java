//******************************************************************************
//
// File:    AList.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.AList
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
import java.lang.reflect.Array;
import java.util.Arrays;

/**
 * Class AList provides a list of items stored in a dynamically-sized array.
 * <P>
 * Operations take constant time unless otherwise specified. <I>n</I> is the
 * number of items stored in the list.
 *
 * @param  <T>  List item data type.
 *
 * @author  Alan Kaminsky
 * @version 22-Feb-2019
 */
public class AList<T>
	implements Streamable
	{

// Hidden data members.

	// Chunk size for growing array.
	private static final int CHUNK = 8;

	// Array of list items.
	T[] item = (T[]) new Object [CHUNK];

	// Number of list items.
	int size = 0;

	// Default searcher and sorter objects.
	private static final Searcher defaultSearcher = new Searcher();
	private static final Sorter defaultSorter = new Sorter();

// Exported helper classes.

	/**
	 * Class AList.Searcher provides an object for customizing the searching
	 * methods in class {@linkplain AList AList}.
	 *
	 * @param  <T>  List item data type.
	 *
	 * @author  Alan Kaminsky
	 * @version 14-Feb-2017
	 */
	public static class Searcher<T>
		{
		/**
		 * Construct a new list searcher object.
		 */
		public Searcher()
			{
			}

		/**
		 * Compare two list items for purposes of searching. This determines the
		 * desired ordering criterion.
		 * <P>
		 * The default implementation of this method casts each item to type
		 * Comparable&lt;T&gt;; calls the <TT>compareTo()</TT> method on item
		 * <TT>a</TT>, passing item <TT>b</TT> as the argument; and returns what
		 * the <TT>compareTo()</TT> method returns. If this is not the desired
		 * behavior, or if the list items cannot be cast to type
		 * Comparable&lt;T&gt;, then override this method in a subclass.
		 *
		 * @param  a  First list item being compared.
		 * @param  b  Second list item being compared.
		 *
		 * @return  A number less than, equal to, or greater than 0 if item
		 *          <TT>a</TT> comes before, is the same as, or comes after item
		 *          <TT>b</TT> in the desired ordering, false otherwise.
		 */
		public int compare
			(T a,
			 T b)
			{
			return ((Comparable<T>)a).compareTo (b);
			}
		}

	/**
	 * Class AList.Sorter provides an object for customizing the sorting methods
	 * in class {@linkplain AList AList}.
	 *
	 * @param  <T>  List item data type.
	 *
	 * @author  Alan Kaminsky
	 * @version 14-Feb-2017
	 */
	public static class Sorter<T>
		{
		/**
		 * Construct a new list sorter object.
		 */
		public Sorter()
			{
			}

		/**
		 * Compare two items in the given list for purposes of sorting. This
		 * determines the sorted order of the list items.
		 * <P>
		 * The default implementation of this method casts each item to type
		 * Comparable&lt;T&gt;; calls the <TT>compareTo()</TT> method on the
		 * item at position <TT>a</TT>, passing the item at position <TT>b</TT>
		 * as the argument; and returns true if the <TT>compareTo()</TT> method
		 * returns a value less than 0. If this is not the desired behavior, or
		 * if the list items cannot be cast to type Comparable&lt;T&gt;, then
		 * override this method in a subclass.
		 *
		 * @param  list  List being sorted.
		 * @param  a     Position of first list item being compared.
		 * @param  b     Position of second list item being compared.
		 *
		 * @return  True if the item at position <TT>a</TT> comes before the
		 *          item at position <TT>b</TT> in the desired ordering, false
		 *          otherwise.
		 */
		public boolean comesBefore
			(AList<T> list,
			 int a,
			 int b)
			{
			return ((Comparable<T>)(list.get (a))).compareTo (list.get (b)) < 0;
			}

		/**
		 * Swap two items in the given list for purposes of sorting.
		 * <P>
		 * The default implementation calls <TT>list.swap(a,b)</TT>. If this is
		 * not the desired behavior, then override this method in a subclass;
		 * for example, to swap the elements of other arrays or lists in
		 * addition to this list.
		 *
		 * @param  list  List being sorted.
		 * @param  a     Position of first list item being swapped.
		 * @param  b     Position of second list item being swapped.
		 */
		protected void swap
			(AList<T> list,
			 int a,
			 int b)
			{
			list.swap (a, b);
			}
		}

// Exported constructors.

	/**
	 * Construct a new list.
	 */
	public AList()
		{
		}

	/**
	 * Construct a new list that is a copy of the given list. The new list's
	 * entries contain the same items in the same order as the given list. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  list  List to copy.
	 */
	public AList
		(AList<T> list)
		{
		this();
		copy (list);
		}

	/**
	 * Construct a new list that is a copy of the given list. The new list's
	 * entries contain the same items in the same order as the given list. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  list  List to copy.
	 */
	public AList
		(DList<T> list)
		{
		this();
		copy (list);
		}

// Exported operations.

	/**
	 * Determine if this list is empty.
	 *
	 * @return  True if this list is empty, false if it isn't.
	 */
	public boolean isEmpty()
		{
		return size == 0;
		}

	/**
	 * Clear this list. Time: <I>O</I>(<I>n</I>).
	 */
	public void clear()
		{
		Arrays.fill (item, 0, size, null);
		size = 0;
		}

	/**
	 * Set this list to a copy of the given list. Afterwards, this list's
	 * entries contain the same items in the same order as the given list. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  list  List to copy.
	 */
	public void copy
		(AList<T> list)
		{
		int newlength = list.size + CHUNK - 1;
		newlength = Math.max (newlength/CHUNK, 1);
		newlength *= CHUNK;
		item = (T[]) new Object [newlength];
		System.arraycopy (list.item, 0, item, 0, list.size);
		size = list.size;
		}

	/**
	 * Set this list to a copy of the given list. Afterwards, this list's
	 * entries contain the same items in the same order as the given list. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  list  List to copy.
	 */
	public void copy
		(DList<T> list)
		{
		clear();
		list.forEachItemDo (new Action<T>()
			{
			public void run (T item)
				{
				addLast (item);
				}
			});
		}

	/**
	 * Returns the number of items in this list.
	 *
	 * @return  Number of items.
	 */
	public int size()
		{
		return size;
		}

	/**
	 * Get the item at the given position in this list.
	 *
	 * @param  p  Position, in the range 0 .. <TT>size()</TT>&minus;1.
	 *
	 * @return  Item stored at position <TT>p</TT>.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>p</TT> is out of bounds.
	 */
	public T get
		(int p)
		{
		if (0 > p || p >= size)
			throw new IndexOutOfBoundsException (String.format
				("AList.get(): p = %d out of bounds", p));
		return item[p];
		}

	/**
	 * Set the item at the given position in this list.
	 *
	 * @param  p  Position, in the range 0 .. <TT>size()</TT>&minus;1.
	 * @param  i  Item to store at position <TT>p</TT>.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>p</TT> is out of bounds.
	 */
	public void set
		(int p,
		 T i)
		{
		if (0 > p || p >= size)
			throw new IndexOutOfBoundsException (String.format
				("AList.set(): p = %d out of bounds", p));
		item[p] = i;
		}

	/**
	 * Swap the items at the given positions in this list.
	 *
	 * @param  p  First item position, in the range 0 ..
	 *            <TT>size()</TT>&minus;1.
	 * @param  q  Second item position, in the range 0 ..
	 *            <TT>size()</TT>&minus;1.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>p</TT> or <TT>q</TT> is out of
	 *     bounds.
	 */
	public void swap
		(int p,
		 int q)
		{
		if (0 > p || p >= size)
			throw new IndexOutOfBoundsException (String.format
				("AList.swap(): p = %d out of bounds", p));
		if (0 > q || q >= size)
			throw new IndexOutOfBoundsException (String.format
				("AList.swap(): q = %d out of bounds", q));
		T tmp = item[p];
		item[p] = item[q];
		item[q] = tmp;
		}

	/**
	 * Swap the item at the given position with the first item in this list.
	 *
	 * @param  p  Item position, in the range 0 .. <TT>size()</TT>&minus;1.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>p</TT> is out of bounds.
	 */
	public void swapFirst
		(int p)
		{
		swap (p, 0);
		}

	/**
	 * Swap the item at the given position with the last item in this list.
	 *
	 * @param  p  Item position, in the range 0 .. <TT>size()</TT>&minus;1.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>p</TT> is out of bounds.
	 */
	public void swapLast
		(int p)
		{
		swap (p, size - 1);
		}

	/**
	 * Get the position of the given item in this list. This involves a linear
	 * scan of the list items starting from the beginning of the list; the
	 * returned value is the position of the first item encountered that is
	 * equal to the given item, as determined by the item's <TT>equals()</TT>
	 * method; if there is no such item, &minus;1 is returned. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  i  Item.
	 *
	 * @return  Position of matching item in the range 0 ..
	 *          <TT>size()</TT>&minus;1, or &minus;1 if none.
	 */
	public int position
		(T i)
		{
		for (int p = 0; p < size; ++ p)
			{
			T item_p = item[p];
			if (item_p == null)
				{
				if (i == null) return p;
				}
			else
				{
				if (item_p.equals (i)) return p;
				}
			}
		return -1;
		}

	/**
	 * Get the position of the first item in this list for which the given
	 * predicate is true. This involves a linear scan of the list items starting
	 * from the beginning of the list; the returned value is the position of the
	 * first item encountered for which the predicate's <TT>test()</TT> method
	 * returns true; if there is no such item, &minus;1 is returned. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  predicate  Predicate.
	 *
	 * @return  Position of matching item in the range 0 ..
	 *          <TT>size()</TT>&minus;1, or &minus;1 if none.
	 */
	public int position
		(Predicate<T> predicate)
		{
		for (int p = 0; p < size; ++ p)
			if (predicate.test (item[p]))
				return p;
		return -1;
		}

	/**
	 * Determine if this list that contains the given item. This involves a
	 * linear scan of the list entries from beginning to end; if an entry is
	 * encountered whose item is equal to the given item, as determined by the
	 * item's <TT>equals()</TT> method, true is returned; otherwise false is
	 * returned. Time: <I>O</I>(<I>n</I>).
	 *
	 * @param  item  Item.
	 *
	 * @return  True if this list contains <TT>item</TT>, false otherwise.
	 */
	public boolean contains
		(T item)
		{
		return position (item) != -1;
		}

	/**
	 * Determine if this list contains an item for which the given predicate is
	 * true. This involves a linear scan of the list entries from beginning to
	 * end; if an entry is encountered for which the predicate's <TT>test()</TT>
	 * method returns true, true is returned; otherwise false is returned. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  predicate  Predicate.
	 *
	 * @return  True if this list contains an item matching <TT>predicate</TT>,
	 *          false otherwise.
	 */
	public boolean contains
		(Predicate<T> predicate)
		{
		return position (predicate) != -1;
		}

	/**
	 * Add the given item to the end of this list. This list's size increases by
	 * 1.
	 *
	 * @param  i  Item.
	 *
	 * @return  Newly added item.
	 */
	public T addLast
		(T i)
		{
		if (size == item.length)
			{
			T[] newitem = (T[]) new Object [item.length + CHUNK];
			System.arraycopy (item, 0, newitem, 0, size);
			item = newitem;
			}
		item[size] = i;
		++ size;
		return i;
		}

	/**
	 * Add the given item to the beginning of this list. Items at position 0 and
	 * beyond are moved forward one position. This list's size increases by 1.
	 * Time: <I>O</I>(<I>n</I>).
	 *
	 * @param  i  Item.
	 *
	 * @return  Newly added item.
	 */
	public T addFirst
		(T i)
		{
		if (size == item.length)
			{
			T[] newitem = (T[]) new Object [item.length + CHUNK];
			System.arraycopy (item, 0, newitem, 1, size);
			item = newitem;
			}
		else
			{
			System.arraycopy (item, 0, item, 1, size);
			}
		item[0] = i;
		++ size;
		return i;
		}

	/**
	 * Add the given item to this list at the given position. Items at position
	 * <TT>p</TT> and beyond are moved forward one position. This list's size
	 * increases by 1. Time: <I>O</I>(<I>n</I>).
	 *
	 * @param  p  Position, in the range 0 .. <TT>size()</TT>.
	 * @param  i  Item.
	 *
	 * @return  Newly added item.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>p</TT> is out of bounds.
	 */
	public T add
		(int p,
		 T i)
		{
		if (0 > p || p > size)
			throw new IndexOutOfBoundsException (String.format
				("AList.add(): p = %d out of bounds", p));
		if (size == item.length)
			{
			T[] newitem = (T[]) new Object [item.length + CHUNK];
			System.arraycopy (item, 0, newitem, 0, p);
			System.arraycopy (item, p, newitem, p + 1, size - p);
			item = newitem;
			}
		else
			{
			System.arraycopy (item, p, item, p + 1, size - p);
			}
		item[p] = i;
		++ size;
		return i;
		}

	/**
	 * Add the given item to the end of this list, then swap the item at the
	 * given position with the item at the end of this list. The newly added
	 * item, which is now at the given position, is returned. This list's size
	 * increases by 1.
	 *
	 * @param  p  Position, in the range 0 .. <TT>size()</TT>.
	 * @param  i  Item.
	 *
	 * @return  Newly added item.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>p</TT> is out of bounds.
	 */
	public T addLastSwap
		(int p,
		 T i)
		{
		if (0 > p || p > size)
			{
			throw new IndexOutOfBoundsException
				("AList.addLastSwap(): p = "+p+" out of bounds");
			}
		addLast (i);
		item[size-1] = item[p];
		item[p] = i;
		return i;
		}

	/**
	 * Remove the item at the end of this list. The removed item is returned.
	 * This list's size decreases by 1. If this list is empty, nothing happens
	 * and null is returned.
	 *
	 * @return  Removed item, or null if this list is empty.
	 */
	public T removeLast()
		{
		if (size == 0) return null;
		-- size;
		T tmp = item[size];
		item[size] = null;
		return tmp;
		}

	/**
	 * Remove the item at the beginning of this list. Items at position 1 and
	 * beyond are moved backward one position. The removed item is returned.
	 * This list's size decreases by 1. If this list is empty, nothing happens
	 * and null is returned. Time: <I>O</I>(<I>n</I>).
	 *
	 * @return  Removed item, or null if this list is empty.
	 */
	public T removeFirst()
		{
		if (size == 0) return null;
		-- size;
		T tmp = item[0];
		System.arraycopy (item, 1, item, 0, size);
		item[size] = null;
		return tmp;
		}

	/**
	 * Remove the item at the given position in this list. Items at position
	 * <TT>p</TT>+1 and beyond are moved backward one position. The removed item
	 * is returned. This list's size decreases by 1. If this list is empty,
	 * nothing happens and null is returned. Time: <I>O</I>(<I>n</I>).
	 *
	 * @param  p  Position, in the range 0 .. <TT>size()</TT>&minus;1.
	 *
	 * @return  Removed item, or null if this list is empty.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if this list is not empty and <TT>p</TT>
	 *     is out of bounds.
	 */
	public T remove
		(int p)
		{
		if (size == 0) return null;
		if (0 > p || p >= size)
			throw new IndexOutOfBoundsException (String.format
				("AList.remove(): p = %d out of bounds", p));
		-- size;
		T tmp = item[p];
		System.arraycopy (item, p + 1, item, p, size - p);
		item[size] = null;
		return tmp;
		}

	/**
	 * Swap the item at the given position with the item at the end of this
	 * list, then remove the item at the end of this list. The removed item,
	 * which used to be at the given position, is returned. This list's size
	 * decreases by 1.
	 *
	 * @param  p  Position, in the range 0 .. <TT>size()</TT>&minus;1.
	 *
	 * @return  Removed item.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>p</TT> is out of bounds.
	 */
	public T swapRemoveLast
		(int p)
		{
		if (0 > p || p >= size)
			throw new IndexOutOfBoundsException (String.format
				("AList.swapRemoveLast(): p = %d out of bounds", p));
		-- size;
		T tmp = item[p];
		item[p] = item[size];
		item[size] = null;
		return tmp;
		}

	/**
	 * Perform the given action on each item in this list. For each item in this
	 * list from beginning to end, the given <TT>action</TT>'s <TT>run()</TT>
	 * method is called, passing in the item. Time: <I>O</I>(<I>n</I>).
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds items to or removes
	 * items from the list, the <TT>forEachItemDo()</TT> method's behavior is
	 * not specified.
	 *
	 * @param  action  Action.
	 */
	public void forEachItemDo
		(Action<T> action)
		{
		for (int p = 0; p < size; ++ p)
			action.run (item[p]);
		}

	/**
	 * Perform the given action on each item in this list and return a result.
	 * For each item in this list from beginning to end, the given
	 * <TT>action</TT>'s <TT>run()</TT> method is called, passing in the item.
	 * After all the list items have been processed, the given <TT>action</TT>'s
	 * <TT>result()</TT> method is called, and its result is returned. Time:
	 * <I>O</I>(<I>n</I>).
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds items to or removes
	 * items from the list, the <TT>forEachItemDo()</TT> method's behavior is
	 * not specified.
	 *
	 * @param  <R>     Result data type.
	 * @param  action  Action.
	 *
	 * @return  Result of processing all the list items.
	 */
	public <R> R forEachItemDo
		(ActionResult<T,R> action)
		{
		for (int p = 0; p < size; ++ p)
			action.run (item[p]);
		return action.result();
		}

	/**
	 * Evaluate the given predicate on, and possibly remove, each item in this
	 * list. For each item in this list from beginning to end, the given
	 * <TT>predicate</TT>'s <TT>test()</TT> method is called, passing in the
	 * item. If the <TT>test()</TT> method returns true, the item is removed
	 * from this list. Time: <I>O</I>(<I>n</I>).
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>predicate</TT> adds items to or removes
	 * items from the list, other than by returning true, the
	 * <TT>removeEachItemIf()</TT> method's behavior is not specified.
	 *
	 * @param  predicate  Predicate.
	 */
	public void removeEachItemIf
		(Predicate<T> predicate)
		{
		int p = 0;
		while (p < size)
			if (predicate.test (item[p]))
				remove (p);
			else
				++ p;
		}

	/**
	 * Store this list's items in the given array. The first item is stored at
	 * index 0, the second item at index 1, and so on. The number of array
	 * elements set is <TT>array.length</TT>. If this list contains fewer than
	 * <TT>array.length</TT> items, the remaining array elements are set to
	 * null. Time: <I>O</I>(<I>n</I>).
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
	 * Store this list's items in the given array. The first item is stored at
	 * index <TT>off</TT>, the second item at index <TT>off</TT>+1, and so on.
	 * The number of array elements set is <TT>len</TT>. If this list contains
	 * fewer than <TT>len</TT> items, the remaining array elements are set to
	 * null. Time: <I>O</I>(<I>n</I>).
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
		System.arraycopy (item, 0, array, off, n);
		if (n < len) Arrays.fill (array, off + n, off + len, null);
		return array;
		}

	/**
	 * Search this unordered list for the given item. The
	 * <TT>searchUnsorted()</TT> method calls the default searcher object's
	 * <TT>compare()</TT> method to compare items for equality only. The default
	 * searcher object is an instance of base class {@linkplain AList.Searcher
	 * AList.Searcher}. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  a  Item to be searched for.
	 *
	 * @return  If an item the same as <TT>a</TT> exists in this list, then the
	 *          position of that item is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public int searchUnsorted
		(T a)
		{
		return searchUnsorted (a, defaultSearcher);
		}

	/**
	 * Search this unordered list for the given item, using the given searcher.
	 * The <TT>searchUnsorted()</TT> method calls the searcher object's
	 * <TT>compare()</TT> method to compare items for equality only. An
	 * <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  a         Item to be searched for.
	 * @param  searcher  Searcher object.
	 *
	 * @return  If an item the same as <TT>a</TT> exists in this list, then the
	 *          position of that item is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public int searchUnsorted
		(T a,
		 Searcher<T> searcher)
		{
		for (int i = 0; i < size; ++ i)
			if (searcher.compare (item[i], a) == 0) return i;
		return -1;
		}

	/**
	 * Search this ordered list for the given item. The <TT>searchSorted()</TT>
	 * method calls the default searcher object's <TT>compare()</TT> method to
	 * compare elements for equality and ordering. The default searcher object
	 * is an instance of base class {@linkplain AList.Searcher AList.Searcher}.
	 * It is assumed that this list is sorted in the order determined by the
	 * searcher's <TT>compare()</TT> method; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log&nbsp;<I>n</I>) binary search algorithm is used.
	 *
	 * @param  a  Item to be searched for.
	 *
	 * @return  If an item the same as <TT>a</TT> exists in this list, then the
	 *          position of that item is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public int searchSorted
		(T a)
		{
		return searchSorted (a, defaultSearcher);
		}

	/**
	 * Search this ordered list for the given item, using the given searcher.
	 * The <TT>searchSorted()</TT> method calls the searcher object's
	 * <TT>compare()</TT> method to compare elements for equality and ordering.
	 * It is assumed that this list is sorted in the order determined by the
	 * searcher's <TT>compare()</TT> method; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log&nbsp;<I>n</I>) binary search algorithm is used.
	 *
	 * @param  a         Item to be searched for.
	 * @param  searcher  Searcher object.
	 *
	 * @return  If an item the same as <TT>a</TT> exists in this list, then the
	 *          position of that item is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public int searchSorted
		(T a,
		 Searcher<T> searcher)
		{
		// Establish loop invariant.
		if (size == 0) return -1;

		int lo = 0;
		int locomp = searcher.compare (item[lo], a);
		if (locomp == 0) return lo;
		else if (locomp > 0) return -1;

		int hi = size - 1;
		int hicomp = searcher.compare (item[hi], a);
		if (hicomp == 0) return hi;
		else if (hicomp < 0) return -1;

		// Loop invariant: item[lo] comes before a; item[hi] comes after a.
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			int midcomp = searcher.compare (item[mid], a);
			if (midcomp == 0)
				{
				return mid;
				}
			else if (midcomp < 0)
				{
				lo = mid;
				locomp = midcomp;
				}
			else
				{
				hi = mid;
				hicomp = midcomp;
				}
			}

		return locomp == 0 ? lo : hicomp == 0 ? hi : -1;
		}

	/**
	 * Search this ordered list for an interval containing the given item. The
	 * <TT>searchInterval()</TT> method calls the default searcher object's
	 * <TT>compare()</TT> method to compare elements for equality and ordering.
	 * The default searcher object is an instance of base class {@linkplain
	 * AList.Searcher AList.Searcher}. It is assumed that this list is sorted in
	 * the order determined by the searcher's <TT>compare()</TT> method;
	 * otherwise, the <TT>searchInterval()</TT> method's behavior is not
	 * specified. An <I>O</I>(log&nbsp;<I>n</I>) binary search algorithm is
	 * used.
	 *
	 * @param  a  Item to be searched for.
	 *
	 * @return
	 *     A position <TT>i</TT> such that <TT>get(i-1)</TT> &le; <TT>a</TT>
	 *     &lt; <TT>get(i)</TT>. If <TT>a</TT> &lt; <TT>get(0)</TT>, then
	 *     <TT>i</TT> = 0 is returned. If <TT>get(size()-1)</TT> &le;
	 *     <TT>a</TT>, then <TT>i</TT> = <TT>size()</TT> is returned. If this
	 *     list is empty, then <TT>i</TT> = &minus;1 is returned.
	 */
	public int searchInterval
		(T a)
		{
		return searchInterval (a, defaultSearcher);
		}

	/**
	 * Search this ordered list for an interval containing the given item, using
	 * the given searcher. The <TT>searchInterval()</TT> method calls the
	 * searcher object's <TT>compare()</TT> method to compare elements for
	 * equality and ordering. It is assumed that this list is sorted in the
	 * order determined by the searcher's <TT>compare()</TT> method; otherwise,
	 * the <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log&nbsp;<I>n</I>) binary search algorithm is used.
	 *
	 * @param  a         Item to be searched for.
	 * @param  searcher  Searcher object.
	 *
	 * @return
	 *     A position <TT>i</TT> such that <TT>get(i-1)</TT> &le; <TT>a</TT>
	 *     &lt; <TT>get(i)</TT>. If <TT>a</TT> &lt; <TT>get(0)</TT>, then
	 *     <TT>i</TT> = 0 is returned. If <TT>get(size()-1)</TT> &le;
	 *     <TT>a</TT>, then <TT>i</TT> = <TT>size()</TT> is returned. If this
	 *     list is empty, then <TT>i</TT> = &minus;1 is returned.
	 */
	public int searchInterval
		(T a,
		 Searcher<T> searcher)
		{
		// Establish loop invariant.
		if (size == 0) return -1;

		int lo = 0;
		if (searcher.compare (item[lo], a) > 0) return 0;

		int hi = size - 1;
		if (searcher.compare (item[hi], a) <= 0) return size;

		// Loop invariant: item[lo] <= a and a < item[hi].
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			if (searcher.compare (item[mid], a) <= 0)
				lo = mid;
			else
				hi = mid;
			}

		return hi;
		}

	/**
	 * Sort this list. The <TT>sort()</TT> method calls the default sorter
	 * object's <TT>comesBefore()</TT> method to determine the sorted order. The
	 * <TT>sort()</TT> method calls the default sorter object's <TT>swap()</TT>
	 * method to swap items during the sort. The default sorter object is an
	 * instance of base class {@linkplain AList.Sorter AList.Sorter}. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 */
	public void sort()
		{
		sort (defaultSorter);
		}

	/**
	 * Sort this list, using the given sorter. The <TT>sort()</TT> method calls
	 * the sorter object's <TT>comesBefore()</TT> method to determine the sorted
	 * order. The <TT>sort()</TT> method calls the sorter object's
	 * <TT>swap()</TT> method to swap items during the sort. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  sorter  Sorter object.
	 */
	public void sort
		(Sorter<T> sorter)
		{
		for (int i = 2; i <= size; ++ i)
			{
			siftUp (i, sorter);
			}
		for (int i = size; i >= 2; -- i)
			{
			sorter.swap (this, 0, i - 1);
			siftDown (i - 1, sorter);
			}
		}

	private void siftUp
		(int c, // 1-based index
		 Sorter<T> sorter)
		{
		int p = c >> 1; // 1-based index
		while (p >= 1)
			{
			if (sorter.comesBefore (this, p - 1, c - 1))
				{
				sorter.swap (this, p - 1, c - 1);
				}
			else
				{
				break;
				}
			c = p;
			p = c >> 1;
			}
		}

	private void siftDown
		(int n, // 1-based index
		 Sorter<T> sorter)
		{
		int p  = 1; // 1-based index
		int ca = 2; // 1-based index
		int cb = 3; // 1-based index
		while (ca <= n)
			{
			if (cb <= n && sorter.comesBefore (this, ca - 1, cb - 1))
				{
				if (sorter.comesBefore (this, p - 1, cb - 1))
					{
					sorter.swap (this, p - 1, cb - 1);
					p = cb;
					}
				else
					{
					break;
					}
				}
			else
				{
				if (sorter.comesBefore (this, p - 1, ca - 1))
					{
					sorter.swap (this, p - 1, ca - 1);
					p = ca;
					}
				else
					{
					break;
					}
				}
			ca = p << 1;
			cb = ca + 1;
			}
		}

	/**
	 * Write this object's fields to the given out stream. The list items are
	 * written using {@link OutStream#writeReference(Object) writeReference()}.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  ClassCastException
	 *     (unchecked exception) Thrown if an item in this list is not
	 *     streamable.
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
	 * Read this object's fields from the given in stream. The list items are
	 * read using {@link InStream#readReference() readReference()}.
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
		int newlength = size + CHUNK - 1;
		newlength = Math.max (newlength/CHUNK, 1);
		newlength *= CHUNK;
		item = (T[]) new Object [newlength];
		for (int i = 0; i < size; ++ i)
			item[i] = (T) in.readReference();
		}

// Unit test main program.

//	/**
//	 * Unit test main program. Creates a list of the given Integer objects, then
//	 * searches the list for object <TT>a</TT>.
//	 * <P>
//	 * Usage: java edu.rit.util.AList <I>items</I> <I>a</I>
//	 * <BR><I>items</I> = List to be searched (int)
//	 * <BR><I>a</I> = Item to be searched for (int)
//	 */
//	public static void main
//		(String[] args)
//		{
//		if (args.length < 1) usage();
//		AList<java.lang.Integer> x = new AList<java.lang.Integer>();
//		for (int i = 0; i < args.length - 1; ++ i)
//			x.addLast (new java.lang.Integer (args[i]));
//		int a = java.lang.Integer.parseInt (args[args.length-1]);
//		System.out.printf ("searchUnsorted(%d) returns %d%n",
//			a, x.searchUnsorted (a));
//		System.out.printf ("searchSorted(%d) returns %d%n",
//			a, x.searchSorted (a));
//		System.out.printf ("searchInterval(%d) returns %d%n",
//			a, x.searchInterval (a));
//		}
//
//	/**
//	 * Print a usage message and exit.
//	 */
//	private static void usage()
//		{
//		System.err.println ("Usage: java edu.rit.util.AList <items> <a>");
//		System.err.println ("<items> = List to be searched (int)");
//		System.err.println ("<a> = Item to be searched for (int)");
//		System.exit (1);
//		}

//	/**
//	 * Unit test main program. Generates a list of <TT>n</TT> random Integer
//	 * objects, then sorts the list.
//	 * <P>
//	 * Usage: java edu.rit.util.AList <I>n</I> <I>seed</I>
//	 * <BR><I>n</I> = Array length
//	 * <BR><I>seed</I> = Random seed
//	 */
//	public static void main
//		(String[] args)
//		{
//		if (args.length != 2) usage();
//		int n = java.lang.Integer.parseInt (args[0]);
//		long seed = java.lang.Long.parseLong (args[1]);
//		AList<java.lang.Integer> x = new AList<java.lang.Integer>();
//		Random prng = new Random (seed);
//		for (int i = 0; i < n; ++ i)
//			x.addLast (new java.lang.Integer (prng.nextInt (100)));
//		for (int i = 0; i < n; ++ i)
//			System.out.printf ("%s ", x.get (i));
//		System.out.println();
//		x.sort();
//		for (int i = 0; i < n; ++ i)
//			System.out.printf ("%s ", x.get (i));
//		System.out.println();
//		}
//
//	/**
//	 * Print a usage message and exit.
//	 */
//	private static void usage()
//		{
//		System.err.println ("Usage: java edu.rit.util.AList <n> <seed>");
//		System.err.println ("<n> = List size");
//		System.err.println ("<seed> = Random seed");
//		System.exit (1);
//		}

	}
