//******************************************************************************
//
// File:    Test07.java
// Package: edu.rit.gpu.test
// Unit:    Class edu.rit.gpu.test.Test07
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

package edu.rit.gpu.test;

import edu.rit.gpu.Gpu;
import edu.rit.pj2.Task;

/**
 * Class Test07 is a unit test main program that prints information about the
 * GPUs on the node.
 * <P>
 * Usage: <TT>java pj2 edu.rit.gpu.test.Test07</TT>
 *
 * @author  Alan Kaminsky
 * @version 11-Apr-2016
 */
public class Test07
	extends Task
	{

	/**
	 * Task main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		System.out.printf ("Gpu.deviceCount() = %d%n", Gpu.deviceCount());
		int N = Gpu.allowedDeviceCount();
		System.out.printf ("Gpu.allowedDeviceCount() = %d%n", N);
		for (int i = 0; i < N; ++ i)
			{
			Gpu gpu = Gpu.gpu();
			System.out.printf ("GPU %d multiprocessors = %d%n",
				i, gpu.getMultiprocessorCount());
			}
		}

	/**
	 * Specify that this task requires one core.
	 */
	protected static int coresRequired()
		{
		return 1;
		}

	/**
	 * Specify that this task requires all GPU accelerators.
	 */
	protected static int gpusRequired()
		{
		return ALL_GPUS;
		}

	}
