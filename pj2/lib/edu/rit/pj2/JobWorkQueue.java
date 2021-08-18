//******************************************************************************
//
// File:    JobWorkQueue.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.JobWorkQueue
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

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.util.Action;
import edu.rit.util.DList;
import java.io.IOException;

/**
 * Class JobWorkQueue provides a distributed queue of work items to be processed
 * by a cluster parallel program (job). There can be only one distributed work
 * queue in a job. The queue is implemented via tuple space.
 * <P>
 * <I>Note:</I> For a work queue in a single-task parallel program, see class
 * {@linkplain WorkQueue}.
 * <P>
 * <B>Programming pattern.</B>
 * To execute a parallel loop over a series of work items in a cluster parallel
 * {@linkplain Job}, follow this pattern in the job's {@link Job#main(String[])
 * main()} method:
 * <PRE>
 * public class MyJob extends Job
 *     {
 *     public void main (String[] args)
 *         {
 *         JobWorkQueue&lt;W&gt; queue = getJobWorkQueue (W.class, workers());
 *         queue.add (<I>initial_work_item</I>);
 *         rule() .task (workers(), <I>worker_task_class</I>);
 *         }
 *     }
 * </PRE>
 * <TT>W</TT> is the data type of the work items; this class must be a subclass
 * of class {@linkplain Tuple}. <TT>queue</TT> is used to access the job's work
 * queue in the job main program. Get a reference to the job's work queue by
 * calling the job's {@link Job#getJobWorkQueue(Class,int) getJobWorkQueue()}
 * method; the arguments are the class of the work items and the number of
 * worker tasks in the job; the latter is typically specified by querying the
 * job's <TT>workers</TT> property. Call the queue's {@link #add(Tuple) add()}
 * method one or more times to add initial work items to the queue. Call the
 * job's {@link Job#rule() rule()} method to set up the job's worker tasks; the
 * number of worker tasks must be the same as was specified in the {@link
 * Job#getJobWorkQueue(Class,int) getJobWorkQueue()} method call.
 * <P>
 * Follow this pattern in the worker task's {@link Task#main(String[]) main()}
 * method:
 * <PRE>
 * public class MyTask extends Task
 *     {
 *     public void main (String[] args)
 *         {
 *         JobWorkQueue&lt;W&gt; queue = getJobWorkQueue (W.class);
 *         W workitem = null;
 *         while ((workitem = queue.remove()) != null)
 *             {
 *             // Code to process workitem
 *             queue.add (<I>another_work_item</I>); // Optional
 *             }
 *         }
 *     }
 * </PRE>
 * <TT>W</TT> is the data type of the work items. <TT>queue</TT> is used to
 * access the job's work queue in the task main program. Get a reference to the
 * job's work queue by calling the task's {@link Task#getJobWorkQueue(Class)
 * getJobWorkQueue()} method; the argument is the class of the work items. Call
 * the queue's {@link #remove() remove()} method to get a work item; process the
 * work item; if needed, call the queue's {@link #add(Tuple) add()} method one
 * or more times to add more work items to the queue; and repeat until the
 * {@link #remove() remove()} method returns null. The task must be single
 * threaded.
 *
 * @param  <W>  Data type of the work items. The work item class must be a
 *              subclass of class {@linkplain Tuple}.
 *
 * @author  Alan Kaminsky
 * @version 20-Mar-2018
 */
public class JobWorkQueue<W extends Tuple>
	{

// Hidden helper classes.

	/**
	 * Tuple class for a work item.
	 */
	private static class WorkItemTuple<W>
		extends Tuple
		{
		public W workitem;

		public WorkItemTuple()
			{
			}

		public WorkItemTuple
			(W workitem)
			{
			this.workitem = workitem;
			}

		public Object clone()
			{
			WorkItemTuple tuple = (WorkItemTuple) super.clone();
			tuple.workitem = ((Tuple)(this.workitem)).clone();
			return tuple;
			}

		public void writeOut
			(OutStream out)
			throws IOException
			{
			out.writeObject (workitem);
			}

		public void readIn
			(InStream in)
			throws IOException
			{
			workitem = (W) in.readObject();
			}
		}

	/**
	 * Tuple class for distributed work queue control.
	 */
	private static class ControlTuple
		extends Tuple
		{
		// Number of worker tasks.
		public int numTasks;

		// Number of worker tasks waiting for a work item.
		public int numWaiting;

		// Number of available work items.
		public int numItems;

		public ControlTuple()
			{
			}

		public ControlTuple
			(int numTasks,
			 int numWaiting,
			 int numItems)
			{
			this.numTasks = numTasks;
			this.numWaiting = numWaiting;
			this.numItems = numItems;
			}

		public void writeOut
			(OutStream out)
			throws IOException
			{
			out.writeInt (numTasks);
			out.writeInt (numWaiting);
			out.writeInt (numItems);
			}

		public void readIn
			(InStream in)
			throws IOException
			{
			numTasks = in.readInt();
			numWaiting = in.readInt();
			numItems = in.readInt();
			}
		}

	/**
	 * Template class to wait for a work item in the distributed work queue.
	 */
	private static class WaitTemplate
		extends ControlTuple
		{
		public <T extends Tuple> Class<T> matchClass()
			{
			return (Class<T>) ControlTuple.class;
			}

		public boolean matchContent
			(Tuple target)
			{
			ControlTuple ct = (ControlTuple) target;
			return ct.numItems > 0 || ct.numWaiting == ct.numTasks;
			}
		}

// Hidden data members.

	private Job job;
	private int numWorkers;
	private DList<W> queue;

	private Task task;

	private WorkItemTuple<W> workTemplate = new WorkItemTuple<W>();
	private ControlTuple controlTemplate = new ControlTuple();
	private ControlTuple waitTemplate = new WaitTemplate();

// Hidden constructors.

	/**
	 * Construct a new job work queue in the given job.
	 *
	 * @param  job         Job that contains the work queue.
	 * @param  numWorkers  Number of worker tasks (&ge; 1), or
	 *                     DEFAULT_WORKERS.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>job</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>numWorkers</TT> &lt; 1.
	 */
	JobWorkQueue
		(Job job,
		 int numWorkers)
		{
		if (numWorkers == Job.DEFAULT_WORKERS) numWorkers = 1;
		if (job == null)
			throw new NullPointerException
				("JobWorkQueue(): job is null");
		if (numWorkers < 1)
			throw new IllegalArgumentException (String.format
				("JobWorkQueue(): numWorkers = %d illegal", numWorkers));
		this.job = job;
		this.numWorkers = numWorkers;
		this.queue = new DList<W>();
		}

	/**
	 * Construct a new job work queue in the given task.
	 *
	 * @param  task  Task that contains the work queue.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>task</TT> is null.
	 */
	JobWorkQueue
		(Task task)
		{
		if (task == null)
			throw new NullPointerException
				("JobWorkQueue(): task is null");
		this.task = task;
		}

// Exported operations.

	/**
	 * Add the given work item to this job work queue.
	 *
	 * @param  workitem  Work item; must be non-null.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>workitem</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void add
		(W workitem)
		throws IOException
		{
		if (workitem == null)
			throw new NullPointerException
				("ClusterWorkQueue.add(): workitem is null");
		if (job != null)
			jobAdd (workitem);
		else if (task != null)
			taskAdd (workitem);
		else
			throw new IllegalStateException ("Shouldn't happen");
		}

	private void jobAdd
		(W workitem)
		{
		if (queue != null)
			queue.addLast (workitem);
		else
			throw new IllegalStateException
				("JobWorkQueue.add(): Cannot add a work item at this point in the job process");
		}

	private void taskAdd
		(W workitem)
		throws IOException
		{
		ControlTuple ct = task.takeTuple (controlTemplate);
		task.putTuple (new WorkItemTuple<W> (workitem));
		task.putTuple (new ControlTuple
			(ct.numTasks, ct.numWaiting, ct.numItems + 1));
		}

	/**
	 * Remove and return the next work item from this job work queue. This
	 * method will block until either a work item is available or there are no
	 * more work items.
	 *
	 * @return  Work item, or null if there are no more work items.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public W remove()
		throws IOException
		{
		if (job != null)
			return jobRemove();
		else if (task != null)
			return taskRemove();
		else
			throw new IllegalStateException ("Shouldn't happen");
		}

	private W jobRemove()
		{
		throw new IllegalStateException
			("JobWorkQueue.remove(): Cannot remove a work item in the job process");
		}

	private W taskRemove()
		throws IOException
		{
		ControlTuple ct;
		ct = task.takeTuple (controlTemplate);
		task.putTuple (new ControlTuple
			(ct.numTasks, ct.numWaiting + 1, ct.numItems));
		ct = task.takeTuple (waitTemplate);
		if (ct.numItems > 0)
			{
			W workitem = task.takeTuple (workTemplate) .workitem;
			task.putTuple (new ControlTuple
				(ct.numTasks, ct.numWaiting - 1, ct.numItems - 1));
			return workitem;
			}
		else
			{
			task.putTuple (ct);
			return null;
			}
		}

// Hidden operations.

	/**
	 * Put initial tuples into tuple space. Called in the job process after the
	 * job's main() method returns.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	void putInitialTuples()
		throws IOException
		{
		queue.forEachItemDo (new Action<W>()
			{
			public void run (W workitem)
				{
				job.putTuple (new WorkItemTuple<W> (workitem));
				}
			});
		job.putTuple (new ControlTuple (numWorkers, 0, queue.size()));
		queue = null;
		}

	}
