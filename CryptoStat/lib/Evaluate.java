//******************************************************************************
//
// File:    Evaluate.java
// Unit:    Class Evaluate
//
// This Java source file is copyright (C) 2017 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This Java source file is part of the CryptoStat Library ("CryptoStat").
// CryptoStat is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; either version 3 of the License, or (at your option) any later
// version.
//
// CryptoStat is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************

import edu.rit.crst.Function;
import edu.rit.crst.FunctionKernel;
import edu.rit.crst.Generator;
import edu.rit.crst.Version;
import edu.rit.gpu.CacheConfig;
import edu.rit.gpu.Gpu;
import edu.rit.gpu.GpuDoubleMatrix;
import edu.rit.gpu.GpuIntArray;
import edu.rit.gpu.GpuIntMatrix;
import edu.rit.pj2.Task;
import edu.rit.util.BigInt;
import edu.rit.util.GpuBigIntArray;
import edu.rit.util.GpuBigIntArray3D;
import edu.rit.util.Instance;
import java.util.Date;

/**
 * Class Evaluate is a GPU accelerated parallel program that evaluates a
 * cryptographic function. It is intended as a cross-check between a CPU
 * implementation and a GPU implementation of the cryptographic function.
 * <P>
 * Usage: <TT>java pj2 Evaluate [-nt=<I>NT</I>] "<I>function</I>"
 * "<I>generator_A</I>" "<I>generator_B</I>"</TT>
 * <P>
 * The <I>function</I> argument is a constructor expression for the {@linkplain
 * Function Function} object that implements the cryptographic function to be
 * analyzed. (For further information about constructor expressions, see class
 * {@linkplain edu.rit.util.Instance edu.rit.util.Instance}.)
 * <P>
 * The <I>generator_A</I> argument is a constructor expression for the
 * {@linkplain Generator Generator} object that generates a series of A input
 * values for the cryptographic function.
 * <P>
 * The <I>generator_B</I> argument is a constructor expression for the
 * {@linkplain Generator Generator} object that generates a series of B input
 * values for the cryptographic function.
 * <P>
 * The Evaluate program does the following:
 * <OL TYPE=1>
 * <P><LI>
 * Generate a series of <I>A</I> inputs for the cryptographic function as
 * specified by the <I>generator_A</I> argument.
 * <P><LI>
 * Generate a series of <I>B</I> inputs for the cryptographic function as
 * specified by the <I>generator_B</I> argument.
 * <P><LI>
 * For each <I>A</I> input, each <I>B</I> input, and each round of the
 * cryptographic function, compute the function's <I>C</I> output on the CPU via
 * the function's <TT>evaluate()</TT> method.
 * <P><LI>
 * For each <I>A</I> input, each <I>B</I> input, and each round of the
 * cryptographic function, compute the function's <I>C</I> output on the GPU via
 * the function's <TT>kernel()</TT> method.
 * <P><LI>
 * Check whether each <I>C</I> output computed on the CPU is the same as each
 * <I>C</I> output computed on the GPU.
 * </OL>
 * <P>
 * The Evaluate program prints each <I>C</I> output computed on the CPU, and
 * flags it if it differs from the value computed on the GPU.
 * <P>
 * If the <TT>-nt=<I>NT</I></TT> flag is specified, <I>NT</I> gives the number
 * of GPU threads per block. If not specified, <I>NT</I> = 768 is used.
 *
 * @author  Alan Kaminsky
 * @version 25-Apr-2017
 */
public class Evaluate
	extends Task
	{

	// Command line arguments.
	String function;
	String generator_A;
	String generator_B;

	// GPU accelerator.
	Gpu gpu;
	int numBlocks;
	int numThreads = 768;

	// Cryptographic function object.
	Function func;

	// Cube bit and regular bit positions for computing superpolys.
	int numCubeBits;
	GpuIntArray cubeBits;
	int numRegBits;
	GpuIntArray regBits;
	int thrInSize;
	GpuBigIntArray thrIn;

	// Cryptographic function inputs and outputs.
	int NA;
	int Asize;
	GpuBigIntArray A;
	int NB;
	int Bsize;
	GpuBigIntArray B;
	int R;
	int Csize;
	GpuBigIntArray3D C;
	BigInt[][][] cpuC;

	/**
	 * Task main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		long t1 = System.currentTimeMillis();

		// Parse command line arguments.
		int argi = 0;
		while (argi < args.length && args[argi].charAt(0) == '-')
			{
			if (args[argi].startsWith ("-nt="))
				{
				try
					{
					numThreads = Integer.parseInt (args[argi].substring (4));
					if (numThreads < 1)
						throw new NumberFormatException();
					}
				catch (NumberFormatException exc)
					{
					error (args[argi] + " illegal");
					}
				}
			else
				usage();
			++ argi;
			}
		if (args.length - argi != 3) usage();
		function = args[argi];
		generator_A = args[argi+1];
		generator_B = args[argi+2];

		// Print provenance.
		System.out.printf ("$ java pj2 Evaluate");
		System.out.printf (" -nt=%d", numThreads);
		System.out.printf (" \"%s\"", function);
		System.out.printf (" \"%s\"", generator_A);
		System.out.printf (" \"%s\"", generator_B);
		System.out.println();
		System.out.printf ("CryptoStat v%d%n", Version.CRYPTOSTAT_VERSION);
		System.out.printf ("Started %s%n", new Date (t1));
		System.out.flush();

		// Set up GPU accelerator.
		gpu = Gpu.gpu();
		gpu.ensureComputeCapability (2, 0);
		numBlocks = gpu.getMultiprocessorCount();
		System.out.printf ("%s%n", gpu);
		System.out.printf ("%d blocks, %d threads per block%n",
			numBlocks, numThreads);
		System.out.flush();

		// Create cryptographic function object.
		func = (Function) Instance.newInstance (function);
		R = func.rounds();
		System.out.printf ("Function = %s%n", func.description());
		System.out.printf ("  A input = %s (%d bits)%n",
			func.A_description(), func.A_bitSize());
		System.out.printf ("  B input = %s (%d bits)%n",
			func.B_description(), func.B_bitSize());
		System.out.printf ("  C output = %s (%d bits)%n",
			func.C_description(), func.C_bitSize());
		System.out.printf ("  Rounds = %d%n", R);
		System.out.flush();

		// Create A input values, upload to GPU.
		Generator gen_A = (Generator) Instance.newInstance (generator_A);
		gen_A.setFunction (func, true);
		A = gen_A.generate();
		NA = A.item.length;
		Asize = A.wordSize();
		System.out.printf ("A input generator = %s%n", gen_A.description());
		System.out.printf ("  Number of inputs = %d%n", NA);
		System.out.printf ("  Initial value = %s%n", A.item[0]);
		System.out.flush();
		A.allocate (gpu);
		A.hostToDev();

		// Create B input values, upload to GPU.
		Generator gen_B = (Generator) Instance.newInstance (generator_B);
		gen_B.setFunction (func, false);
		B = gen_B.generate();
		NB = B.item.length;
		Bsize = B.wordSize();
		System.out.printf ("B input generator = %s%n", gen_B.description());
		System.out.printf ("  Number of inputs = %d%n", NB);
		System.out.printf ("  Initial value = %s%n", B.item[0]);
		System.out.flush();
		B.allocate (gpu);
		B.hostToDev();

		// Create C output values, upload to GPU.
		C = new GpuBigIntArray3D (func.C_bitSize(), NA, NB, R);
		Csize = C.wordSize();
		C.allocate (gpu);
		C.hostToDev();

		// Print function inputs.
		System.out.printf ("%n======== A INPUTS ========%n");
		System.out.printf ("#\tA Input Value%n");
		System.out.flush();
		for (int i = 0; i < NA; ++ i)
			{
			System.out.printf ("%d\t%s%n", i, A.item[i]);
			System.out.flush();
			}
		System.out.printf ("%n======== B INPUTS ========%n");
		System.out.printf ("#\tB Input Value%n");
		System.out.flush();
		for (int i = 0; i < NB; ++ i)
			{
			System.out.printf ("%d\t%s%n", i, B.item[i]);
			System.out.flush();
			}

		// Evaluate cryptographic function on the CPU.
		cpuC = new BigInt [NA] [NB] [R];
		for (int i = 0; i < NA; ++ i)
			for (int j = 0; j < NB; ++ j)
				for (int k = 0; k < R; ++ k)
					cpuC[i][j][k] = new BigInt (func.C_bitSize());
		for (int i = 0; i < NA; ++ i)
			for (int j = 0; j < NB; ++ j)
				func.evaluate (A.item[i], B.item[j], cpuC[i][j]);

		// Evaluate cryptographic function in parallel on the GPU.
		FunctionKernel kernel = func.functionKernel (gpu);
		kernel.setGridDim (numBlocks);
		kernel.setBlockDim (numThreads);
		kernel.setCacheConfig (CacheConfig.CU_FUNC_CACHE_PREFER_L1);
		kernel.evaluateFunction
			(NA, Asize, A.array(),
			 NB, Bsize, B.array(),
			 R,  Csize, C.array());

		// Print function outputs.
		C.devToHost();
		System.out.printf ("%n======== C OUTPUTS ========%n");
		System.out.printf ("A#\tB#\tRound\tC Output Value%n");
		System.out.flush();
		for (int i = 0; i < NA; ++ i)
			for (int j = 0; j < NB; ++ j)
				for (int r = 0; r < R; ++ r)
					{
					System.out.printf ("%d\t%d\t%d\t%s%n",
						i, j, r, cpuC[i][j][r]);
					if (! cpuC[i][j][r].eq (C.item[i][j][r]))
						System.out.printf ("\t\t**GPU**\t%s%n",
							C.item[i][j][r]);
					System.out.flush();
					}

		long t2 = System.currentTimeMillis();
		System.out.printf ("%nFinished %s%n", new Date (t2));
		System.out.printf ("%d msec%n", t2 - t1);
		}

	/**
	 * Print an error message and exit.
	 */
	private static void error
		(String msg)
		{
		System.err.printf ("Evaluate: %s%n", msg);
		usage();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 Evaluate [-nt=<NT>] \"<function>\" \"<generator_A>\" \"<generator_B>\"");
		System.err.println ("<NT> = Number of GPU threads per block (default 768)");
		System.err.println ("<function> = Cryptographic function constructor expression");
		System.err.println ("<generator_A> = A input generator constructor expression");
		System.err.println ("<generator_B> = B input generator constructor expression");
		terminate (1);
		}

	/**
	 * This program requires one CPU core.
	 */
	protected static int coresRequired()
		{
		return 1;
		}

	/**
	 * This program requires one GPU accelerator.
	 */
	protected static int gpusRequired()
		{
		return 1;
		}

	}
