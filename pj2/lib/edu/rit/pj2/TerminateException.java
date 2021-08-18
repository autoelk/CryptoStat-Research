//******************************************************************************
//
// File:    TerminateException.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.TerminateException
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

package edu.rit.pj2;

/**
 * Class TerminateException causes the PJ2 {@linkplain edu.rit.pj2.Job Job} or
 * {@linkplain edu.rit.pj2.Task Task} that throws it to terminate. No error message or exception stack trace is printed. An exit code specified as a constructor argument is returned to the operating system.
 *
 * @author  Alan Kaminsky
 * @version 27-May-2016
 */
public class TerminateException
	extends RuntimeException
	{

	/**
	 * Exit code to be returned to the operating system.
	 */
	public final int exitCode;

	/**
	 * Construct a new terminate exception with an exit code of 0.
	 */
	public TerminateException()
		{
		this (0);
		}

	/**
	 * Construct a new terminate exception with the given exit code.
	 *
	 * @param  exitCode  Exit code to be returned to the operating system.
	 */
	public TerminateException
		(int exitCode)
		{
		super();
		this.exitCode = exitCode;
		}

	}
