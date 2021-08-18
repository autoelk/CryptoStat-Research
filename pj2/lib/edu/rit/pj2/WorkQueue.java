//******************************************************************************
//
// File:    WorkQueue.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.WorkQueue
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

package edu.rit.pj2;

import edu.rit.util.DList;

/**
 * Class WorkQueue provides a queue of work items to be performed by an
 * object parallel for loop in a multicore parallel program. For further
 * information, see class {@linkplain ObjectParallelForLoop} and class
 * {@linkplain ObjectLoop}.
 * <P>
 * <I>Note:</I> For a work queue in a cluster parallel program, see class
 * {@linkplain JobWorkQueue}.
 *
 * @param  <W>  Data type of the work items.
 *
 * @author  Alan Kaminsky
 * @version 20-Mar-2018
 */
public class WorkQueue<W>
	{

// Hidden data members.

	private DList<W> queue = new DList<W>();
	private int threads = Integer.MAX_VALUE;
	private int blocked = 0;

// Exported constructors.

	/**
	 * Construct a new work queue.
	 */
	public WorkQueue()
		{
		super();
		}

// Exported operations.

	/**
	 * Add the given work item to this work queue.
	 *
	 * @param  workitem  Work item; must be non-null.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>workitem</TT> is null.
	 */
	public synchronized void add
		(W workitem)
		{
		if (workitem == null)
			throw new NullPointerException
				("WorkQueue.add(): workitem is null");
		queue.addLast (workitem);
		notifyAll();
		}

	/**
	 * Remove and return the next work item from this work queue. This method
	 * will block until either a work item is available or there are no more
	 * work items.
	 *
	 * @return  Work item, or null if there are no more work items.
	 *
	 * @exception  InterruptedException
	 *     Thrown if the calling thread is interrupted while blocked in this
	 *     method.
	 */
	public synchronized W remove()
		throws InterruptedException
		{
		++ blocked;
		notifyAll();
		while (queue.isEmpty() && blocked < threads)
			wait();
		if (! queue.isEmpty())
			{
			-- blocked;
			notifyAll();
			return queue.first().remove().item();
			}
		else
			return null;
		}

// Hidden operations.

	/**
	 * Get this work queue's <TT>threads</TT> property.
	 *
	 * @return  <TT>threads</TT> property.
	 */
	synchronized int threads()
		{
		return threads;
		}

	/**
	 * Set this work queue's <TT>threads</TT> property.
	 *
	 * @param  threads  <TT>threads</TT> property.
	 */
	synchronized void threads
		(int threads)
		{
		this.threads = threads;
		notifyAll();
		}

	}
