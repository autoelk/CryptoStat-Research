//******************************************************************************
//
// File:    NativeTask.java
// Package: ---
// Unit:    Class NativeTask
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

import edu.rit.pj2.Task;
import java.io.IOException;
import java.util.Map;

/**
 * Class NativeTask lets the Parallel Java 2 (PJ2) middleware run a non-PJ2
 * executable program as a PJ2 {@linkplain edu.rit.pj2.Task Task}. If the PJ2
 * {@linkplain edu.rit.pj2.tracker.Tracker Tracker} daemon is present, the
 * non-PJ2 executable program will run via the Tracker's job queue.
 * <P>
 * Usage: <TT>java pj2 [ <I>pj2_options</I> ] NativeTask <I>executable</I> [
 * <I>args</I> ... ]</TT>
 * <P>
 * Refer to the {@link pj2 pj2} launcher program's documentation for
 * descriptions of the <TT><I>pj2_options</I></TT> that may be specified; such
 * as <TT>cores</TT> for the number of CPU cores required, <TT>gpus</TT> for the
 * number of GPU accelerators required, <TT>timelimit</TT> for a time limit on
 * the program run, <TT>debug</TT> to turn on various debug flags, and others.
 * <P>
 * <TT><I>executable</I></TT> is the full path name of the executable program to
 * be run, followed by the command line arguments for that program if any.
 * <P>
 * In the executable program's environment, certain environment variables are
 * set, as follows. For further information, refer to class {@linkplain
 * edu.rit.pj2.Task Task}'s documentation.
 * <P>
 * &nbsp;&nbsp;&bull;&nbsp;&nbsp; <TT>PJ2_THREADS</TT> = the task's {@link edu.rit.pj2.Task#threads() threads()} property
 * <BR>&nbsp;&nbsp;&bull;&nbsp;&nbsp; <TT>PJ2_ACTUAL_THREADS</TT> = the task's {@link edu.rit.pj2.Task#actualThreads() actualThreads()} property
 * <BR>&nbsp;&nbsp;&bull;&nbsp;&nbsp; <TT>PJ2_CORES</TT> = the task's {@link edu.rit.pj2.Task#cores() cores()} property
 * <BR>&nbsp;&nbsp;&bull;&nbsp;&nbsp; <TT>PJ2_GPUS</TT> = the task's {@link edu.rit.pj2.Task#gpus() gpus()} property
 *
 * @author  Alan Kaminsky
 * @version 06-Dec-2017
 */
public class NativeTask
	extends Task
	{

	/**
	 * Task main program.
	 */
	public void main
		(String[] args)
		{
		// Validate command line arguments.
		if (args.length < 1)
			{
			System.err.println ("Usage: java pj2 [<pj2_options>] NativeTask <executable> [<args> ...]");
			terminate (1);
			}

		// Set up subprocess.
		ProcessBuilder builder = new ProcessBuilder (args);
		Map<String,String> envmap = builder.environment();
		envmap.put ("PJ2_THREADS", ""+threads());
		envmap.put ("PJ2_ACTUAL_THREADS", ""+actualThreads());
		envmap.put ("PJ2_CORES", ""+cores());
		envmap.put ("PJ2_GPUS", ""+gpus());
		builder.inheritIO();

		// Start subprocess.
		Process proc = null;
		try
			{
			proc = builder.start();
			}
		catch (IOException exc)
			{
			System.err.printf ("NativeTask: Cannot start executable program%n");
			System.err.printf ("%s: %s%n", exc.getClass().getName(),
				exc.getMessage());
			terminate (1);
			}

		// Wait for subprocess to finish. Return its exit value.
		try
			{
			terminate (proc.waitFor());
			}
		catch (InterruptedException exc)
			{
			throw new RuntimeException ("Shouldn't happen", exc);
			}
		}

	}
