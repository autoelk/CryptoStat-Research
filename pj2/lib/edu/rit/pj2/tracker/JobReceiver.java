//******************************************************************************
//
// File:    JobReceiver.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.JobReceiver
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

package edu.rit.pj2.tracker;

import edu.rit.pj2.Tuple;
import java.io.EOFException;
import java.net.SocketException;

/**
 * Class JobReceiver provides a thread that receives messages from a Job. A job
 * receiver must be attached to a {@linkplain Proxy Proxy} before the job
 * receiver can be used.
 *
 * @author  Alan Kaminsky
 * @version 03-Feb-2017
 */
public class JobReceiver
	extends Receiver
	{

// Hidden data members.

	private JobRef job;

// Exported constructors.

	/**
	 * Construct a new job receiver. Failures are reported to the given
	 * listener. Incoming messages invoke methods on the given Job.
	 *
	 * @param  listener  Receiver listener.
	 * @param  job       Job.
	 */
	public JobReceiver
		(ReceiverListener listener,
		 JobRef job)
		{
		super (listener);
		this.job = job;
		opcode = Opcode.JOB;
		}

// Exported operations.

	/**
	 * Run this job receiver thread.
	 */
	public void run()
		{
		byte opcode;
		long jobID, taskID, requestID;
		int[] devnum;
		boolean runInJobProcess, blocking, taking;
		String name;
		Tuple tuple, template;
		Throwable exc;
		int copies, stream, len;
		byte[] data = new byte [256];
//int msgnum;

		try
			{
			// Repeatedly read a message and invoke a method on the job.
			for (;;)
				{
//msgnum = nextMsgNum();
				opcode = in.readByte();
				switch (opcode)
					{
					case Opcode.JOBREF_JOB_LAUNCHED:
//printDebug (msgnum, "JOBREF_JOB_LAUNCHED");
						jobID = in.readLong();
						job.jobLaunched (jobID);
						break;
					case Opcode.JOBREF_JOB_STARTED:
//printDebug (msgnum, "JOBREF_JOB_STARTED");
						job.jobStarted();
						break;
					case Opcode.JOBREF_TASK_LAUNCHING:
//printDebug (msgnum, "JOBREF_TASK_LAUNCHING");
						taskID = in.readLong();
						devnum = in.readIntArray();
						runInJobProcess = in.readBoolean();
						job.taskLaunching (taskID, devnum, runInJobProcess);
						devnum = null;
						break;
					case Opcode.JOBREF_TASK_LAUNCHED:
//printDebug (msgnum, "JOBREF_TASK_LAUNCHED");
						taskID = in.readLong();
						name = in.readString();
						job.taskLaunched ((BackendRef)sender, taskID, name);
						name = null;
						break;
					case Opcode.JOBREF_TAKE_TUPLE:
//printDebug (msgnum, "JOBREF_TAKE_TUPLE");
						taskID = in.readLong();
						requestID = in.readLong();
						template = (Tuple) in.readObject();
						blocking = in.readBoolean();
						taking = in.readBoolean();
						job.takeTuple (taskID, requestID, template, blocking,
							taking);
						template = null;
						break;
					case Opcode.JOBREF_WRITE_TUPLE:
//printDebug (msgnum, "JOBREF_WRITE_TUPLE");
						taskID = in.readLong();
						tuple = (Tuple) in.readObject();
						copies = in.readInt();
						job.writeTuple (taskID, tuple, copies);
						tuple = null;
						break;
					case Opcode.JOBREF_TASK_FINISHED:
//printDebug (msgnum, "JOBREF_TASK_FINISHED");
						taskID = in.readLong();
						job.taskFinished (taskID);
						break;
					case Opcode.JOBREF_TASK_FAILED:
//printDebug (msgnum, "JOBREF_TASK_FAILED");
						taskID = in.readLong();
						exc = (Throwable) in.readObject();
						job.taskFailed (taskID, exc);
						exc = null;
						break;
					case Opcode.JOBREF_HEARTBEAT_FROM_TRACKER:
//printDebug (msgnum, "JOBREF_HEARTBEAT_FROM_TRACKER");
						job.heartbeatFromTracker();
						break;
					case Opcode.JOBREF_HEARTBEAT_FROM_TASK:
//printDebug (msgnum, "JOBREF_HEARTBEAT_FROM_TASK");
						taskID = in.readLong();
						job.heartbeatFromTask (taskID);
						break;
					case Opcode.JOBREF_WRITE_STANDARD_STREAM:
//printDebug (msgnum, "JOBREF_WRITE_STANDARD_STREAM");
						stream = in.readInt();
						len = in.readInt();
						if (len > data.length)
							data = new byte [len];
						in.readByteArray (data, 0, len);
						job.writeStandardStream (stream, len, data);
						break;
					case Opcode.SHUTDOWN:
//printDebug (msgnum, "SHUTDOWN");
						throw new EOFException();
					default:
						throw new IllegalArgumentException (String.format
							("JobReceiver.run(): Opcode = %d illegal",
							 opcode));
					}
//printDebug (msgnum, "processed");
				}
			}

		catch (EOFException exc2)
			{
			proxy.farEndShutdown();
			}
		catch (SocketException exc2)
			{
			proxy.farEndShutdown();
			}
		catch (Throwable exc2)
			{
			listener.receiverFailed (this, exc2);
			}
		}

// For debugging.

//	private static int msgnum = 0;
//	private static long t1 = System.currentTimeMillis();
//
//	private static synchronized int nextMsgNum()
//		{
//		return ++ msgnum;
//		}
//
//	private static synchronized void printDebug
//		(int msgnum,
//		 String msg)
//		{
//		System.out.printf ("%d %d %s%n",
//			(System.currentTimeMillis() - t1)/1000, msgnum, msg);
//		System.out.flush();
//		}

	}
