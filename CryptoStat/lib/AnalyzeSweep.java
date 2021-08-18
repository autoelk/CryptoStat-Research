//******************************************************************************
//
// File:    AnalyzeSweep.java
// Unit:    Class AnalyzeSweep
//
// This Java source file is copyright (C) 2018 by Alan Kaminsky. All rights
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

import edu.rit.crst.BitGroup;
import edu.rit.crst.Function;
import edu.rit.crst.FunctionKernel;
import edu.rit.crst.Generator;
import edu.rit.crst.Test;
import edu.rit.crst.TestKernel;
import edu.rit.crst.TestResult;
import edu.rit.crst.Version;
import edu.rit.gpu.CacheConfig;
import edu.rit.gpu.Gpu;
import edu.rit.gpu.GpuIntMatrix;
import edu.rit.gpu.GpuStructMatrix;
import edu.rit.pj2.Task;
import edu.rit.util.AList;
import edu.rit.util.GpuBigIntArray;
import edu.rit.util.GpuBigIntArray3D;
import edu.rit.util.Instance;
import java.io.File;
import java.util.Date;
import java.util.Scanner;

/**
 * Class AnalyzeSweep is a GPU accelerated parallel program that analyzes a
 * cryptographic function by computing odds ratios. The program automatically
 * sweeps through all possible combinations of the parameters.
 * <P>
 * Usage: <TT>java pj2 AnalyzeSweep [-nt=<I>NT</I>] "<I>function</I>"
 * <I>A_file</I> <I>B_file</I> "<I>test</I>" <I>bitGroup_file</I>
 * [ <I>A_start</I> <I>B_start</I> <I>BG_start</I> <I>NRR</I> ]</TT>
 * <P>
 * The <I>function</I> argument is a constructor expression for the {@linkplain
 * Function Function} object that implements the cryptographic function to be
 * analyzed. (For further information about constructor expressions, see class
 * {@linkplain edu.rit.util.Instance edu.rit.util.Instance}.)
 * <P>
 * The <I>A_file</I> argument is the name of a file, each line of which consists
 * of a constructor expression for the {@linkplain Generator Generator} object
 * that generates a series of A input values for the cryptographic function.
 * <P>
 * The <I>B_file</I> argument is the name of a file, each line of which consists
 * of a constructor expression for the {@linkplain Generator Generator} object
 * that generates a series of B input values for the cryptographic function.
 * <P>
 * The <I>test</I> argument is a constructor expression for the {@linkplain Test
 * Test} object that computes a series of odds ratios from the outputs of the
 * cryptographic function.
 * <P>
 * The <I>bitGroup_file</I> argument is the name of a file, each line of which
 * consists of a constructor expression for the {@linkplain BitGroup BitGroup}
 * object that specifies a series of output bit groups to be tested.
 * <P>
 * The AnalyzeSweep program analyzes the cryptographic function specified by the
 * <I>function</I> argument, as follows:
 * <OL TYPE=1>
 * <P><LI>
 * For each constructor expression in the <I>A_file</I>, create a {@linkplain
 * Generator Generator} object and use it to generate a series of <I>A</I>
 * inputs for the cryptographic function.
 * <P><LI>
 * For each constructor expression in the <I>B_file</I>, create a {@linkplain
 * Generator Generator} object and use it to generate a series of <I>B</I>
 * inputs for the cryptographic function.
 * <P><LI>
 * For each constructor expression in the <I>bitGroup_file</I>, create a
 * {@linkplain BitGroup BitGroup} object.
 * <P><LI>
 * For each possible combination of an <I>A</I> input series, a <I>B</I> input
 * series, and a bit group object, carry out an analysis of the cryptographic
 * function using the {@linkplain Test Test} object as described in the
 * {@linkplain Analyze Analyze} program, and report the number of nonrandom
 * rounds detected. The maximum number of nonrandom rounds detected so far is
 * also printed.
 * </OL>
 * <P>
 * If the optional <TT>-nt=<I>NT</I></TT> flag is specified, <I>NT</I> gives the
 * number of GPU threads per block. If not specified, <I>NT</I> = 768 is used.
 * <P>
 * If the optional <I>A_start</I>, <I>B_start</I>, and <I>BG_start</I> arguments
 * are specified, the program uses those as the starting indexes for the
 * <I>A</I> input series, <I>B</I> input series, and bit group, respectively,
 * and proceeds from there. This is intended for resuming an aborted analysis
 * sweep in the middle. If these arguments are omitted, the analysis sweep
 * starts at indexes 0, 0, 0.
 * <P>
 * If the optional <I>NRR</I> argument is specified, the program uses that as
 * the maximum number of nonrandom rounds found so far. This is intended for
 * resuming an aborted analysis sweep in the middle. If this argument is
 * omitted, <I>NRR</I> starts at 0.
 *
 * @author  Alan Kaminsky
 * @version 22-Feb-2018
 */
public class AnalyzeSweep
	extends Task
	{

	// Command line arguments.
	String functionSpec;
	File A_file;
	File B_file;
	String testSpec;
	File bitGroup_file;
	int A_start;
	int B_start;
	int BG_start;

	// GPU accelerator.
	Gpu gpu;
	int numBlocks;
	int numThreads = 768;

	// Cryptographic function object.
	Function func;
	int R;

	// Cryptographic function inputs and outputs.
	GpuBigIntArray[] A;
	GpuBigIntArray[] B;
	int[] NA;
	int[] NB;
	int maxNA;
	int maxNB;
	GpuBigIntArray3D C;
	int Asize;
	int Bsize;
	int Csize;

	// Odds ratio test object and bit group objects.
	Test test;
	BitGroup[] bitGroups;
	int[] NBG;
	int maxNBG;

	// Log Bayes factors.
	int NINPUT;
	int maxNINPUT;
	TestResult[][][] testResult;
	GpuStructMatrix<TestResult> devTestResult;
	GpuIntMatrix[] pos;

	// Number of nonrandom rounds.
	int NRR;

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
		if (args.length - argi < 5) usage();
		functionSpec = args[argi++];
		A_file = new File (args[argi++]);
		B_file = new File (args[argi++]);
		testSpec = args[argi++];
		bitGroup_file = new File (args[argi++]);
		if (args.length - argi == 4)
			{
			A_start = Integer.parseInt (args[argi++]);
			B_start = Integer.parseInt (args[argi++]);
			BG_start = Integer.parseInt (args[argi++]);
			NRR = Integer.parseInt (args[argi++]);
			}
		if (args.length - argi != 0) usage();

		// Print provenance.
		System.out.printf ("$ java pj2 AnalyzeSweep");
		System.out.printf (" -nt=%d", numThreads);
		System.out.printf (" \"%s\"", functionSpec);
		System.out.printf (" %s", A_file);
		System.out.printf (" %s", B_file);
		System.out.printf (" \"%s\"", testSpec);
		System.out.printf (" %s", bitGroup_file);
		if (A_start != 0 || B_start != 0 || BG_start != 0 || NRR != 0)
			System.out.printf (" %d %d %d %d", A_start, B_start, BG_start, NRR);
		System.out.println();
		System.out.printf ("CryptoStat v%d%n", Version.CRYPTOSTAT_VERSION);
		System.out.printf ("Started %s%n", new Date (t1));
		System.out.printf ("Directory %s%n", System.getProperty ("user.dir"));
		System.out.printf ("CLASSPATH %s%n", System.getenv ("CLASSPATH"));
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
		func = (Function) Instance.newInstance (functionSpec);
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
		System.out.printf ("%n======== A INPUT GENERATORS ========%n");
		System.out.printf ("A#\tConstructor%n");
		System.out.flush();
		AList<String> ctorList = new AList<String>();
		Scanner scanner = new Scanner (A_file);
		while (scanner.hasNextLine())
			{
			String ctor = scanner.nextLine();
			System.out.printf ("%d\t%s%n", ctorList.size(), ctor);
			System.out.flush();
			ctorList.addLast (ctor);
			}
		scanner.close();
		A = new GpuBigIntArray [ctorList.size()];
		NA = new int [A.length];
		maxNA = 0;
		for (int i = 0; i < A.length; ++ i)
			{
			Generator gen = (Generator) Instance.newInstance (ctorList.get (i));
			gen.setFunction (func, true);
			A[i] = gen.generate();
			NA[i] = A[i].item.length;
			maxNA = Math.max (maxNA, NA[i]);
			A[i].allocate (gpu);
			A[i].hostToDev();
			}
		Asize = A[0].wordSize();

		// Create B input values, upload to GPU.
		System.out.printf ("%n======== B INPUT GENERATORS ========%n");
		System.out.printf ("B#\tConstructor%n");
		System.out.flush();
		ctorList.clear();
		scanner = new Scanner (B_file);
		while (scanner.hasNextLine())
			{
			String ctor = scanner.nextLine();
			System.out.printf ("%d\t%s%n", ctorList.size(), ctor);
			System.out.flush();
			ctorList.addLast (ctor);
			}
		scanner.close();
		B = new GpuBigIntArray [ctorList.size()];
		NB = new int [B.length];
		maxNB = 0;
		for (int i = 0; i < B.length; ++ i)
			{
			Generator gen = (Generator) Instance.newInstance (ctorList.get (i));
			gen.setFunction (func, false);
			B[i] = gen.generate();
			NB[i] = B[i].item.length;
			maxNB = Math.max (maxNB, NB[i]);
			B[i].allocate (gpu);
			B[i].hostToDev();
			}
		Bsize = B[0].wordSize();

		// Create C output values, allocate on GPU.
		C = new GpuBigIntArray3D (func.C_bitSize(), maxNA, maxNB, R);
		C.allocate (gpu);
		Csize = C.wordSize();

		// Create test and bit group objects.
		test = (Test) Instance.newInstance (testSpec);
		System.out.printf ("%n======== ODDS RATIO TEST AND BIT GROUPS ========%n");
		System.out.printf ("Odds ratio test = %s%n", test.description());
		System.out.printf ("BG#\tConstructor%n");
		System.out.flush();
		ctorList.clear();
		scanner = new Scanner (bitGroup_file);
		while (scanner.hasNextLine())
			{
			String ctor = scanner.nextLine();
			System.out.printf ("%d\t%s%n", ctorList.size(), ctor);
			System.out.flush();
			ctorList.addLast (ctor);
			}
		scanner.close();
		bitGroups = new BitGroup [ctorList.size()];
		NBG = new int [bitGroups.length];
		maxNBG = 0;
		pos = new GpuIntMatrix [bitGroups.length];
		for (int i = 0; i < bitGroups.length; ++ i)
			{
			bitGroups[i] = (BitGroup) Instance.newInstance (ctorList.get (i));
			bitGroups[i].setFunction (func);
			NBG[i] = bitGroups[i].count();
			maxNBG = Math.max (maxNBG, NBG[i]);
			pos[i] = gpu.getIntMatrix (NBG[i], bitGroups[i].bitSize());
			bitGroups[i].bitGroups (pos[i].item);
			pos[i].hostToDev();
			}

		// Set up storage for log Bayes factors.
		maxNINPUT = maxNA + maxNB;
		testResult = new TestResult [maxNINPUT] [R] [maxNBG];
		for (int i = 0; i < maxNINPUT; ++ i)
			for (int j = 0; j < R; ++ j)
				for (int k = 0; k < maxNBG; ++ k)
					testResult[i][j][k] = new TestResult();
		devTestResult = gpu.getStructMatrix (TestResult.class, R, maxNBG);
		for (int j = 0; j < R; ++ j)
			for (int k = 0; k < maxNBG; ++ k)
				devTestResult.item[j][k] = new TestResult();

		// Prepare to print results.
		System.out.printf ("%n======== NONRANDOM ROUNDS (NRR) ========%n");
		System.out.printf ("A#\tB#\tBG#\tNRR\tMaxNRR\tmsec%n");
		System.out.flush();

		// Set up GPU kernel for evaluating cryptographic function.
		FunctionKernel fkernel = func.functionKernel (gpu);
		fkernel.setGridDim (numBlocks);
		fkernel.setBlockDim (numThreads);
		fkernel.setCacheConfig (CacheConfig.CU_FUNC_CACHE_PREFER_L1);

		// Set up GPU kernel for computing odds ratio tests.
		TestKernel tkernel = test.kernel (gpu);
		tkernel.setGridDim (numBlocks);
		tkernel.setBlockDim (numThreads);
		tkernel.setCacheConfig (CacheConfig.CU_FUNC_CACHE_PREFER_L1);

		// Iterate over all combinations of A input series, B input series, and
		// bit groups.
		long t3 = System.currentTimeMillis();
		int Aidx = A_start;
		int Bidx = B_start;
		int BGidx = BG_start;
		boolean evaluate = true;
		while (Aidx < A.length)
			{
			// Sanity check on input series lengths.
			if (NA[Aidx] <= 1 && NB[Bidx] <= 1)
				{
				System.out.printf ("%d\t%d\tNA = %d and NB = %d illegal%n",
					Aidx, Bidx, NA[Aidx], NB[Bidx]);
				System.out.flush();
				}
			else
				{
				// Evaluate cryptographic function in parallel.
				if (evaluate)
					{
//System.out.printf ("Evaluate cryptographic function, Aidx=%d, Bidx=%d, BGidx=%d, NA[Aidx]=%d, NB[Bidx]=%d, NBG[BGidx]=%d%n", Aidx, Bidx, BGidx, NA[Aidx], NB[Bidx], NBG[BGidx]); System.out.flush();
					fkernel.evaluateFunction
						(NA[Aidx], Asize, A[Aidx].array(),
						 NB[Bidx], Bsize, B[Bidx].array(),
						 R,  Csize, C.array());
					evaluate = false;
					}

				// Compute odds ratios for each A input and all B inputs.
				if (NB[Bidx] > 1)
					{
//System.out.printf ("Compute odds ratios for each A input and all B inputs, Aidx=%d, Bidx=%d, BGidx=%d, NA[Aidx]=%d, NB[Bidx]=%d, NBG[BGidx]=%d%na =", Aidx, Bidx, BGidx, NA[Aidx], NB[Bidx], NBG[BGidx]); System.out.flush();
					for (int a = 0; a < NA[Aidx]; ++ a)
						{
//System.out.printf (" %d", a); System.out.flush();
						tkernel.computeTest
							(NA[Aidx], NB[Bidx], R,  C.wordSize(), C.array(),
							 NBG[BGidx], devTestResult,
							 bitGroups[BGidx].bitSize(), pos[BGidx], a, -1);
						recordLbf (testResult[a], BGidx);
						}
//System.out.println(); System.out.flush();
					}
				else
					{
					for (int a = 0; a < NA[Aidx]; ++ a)
						zeroLbf (testResult[a], BGidx);
					}

				// Compute odds ratios for each B input and all A inputs.
				if (NA[Aidx] > 1)
					{
//System.out.printf ("Compute odds ratios for each B input and all A inputs, Aidx=%d, Bidx=%d, BGidx=%d, NA[Aidx]=%d, NB[Bidx]=%d, NBG[BGidx]=%d%nb =", Aidx, Bidx, BGidx, NA[Aidx], NB[Bidx], NBG[BGidx]); System.out.flush();
					for (int b = 0; b < NB[Bidx]; ++ b)
						{
//System.out.printf (" %d", b); System.out.flush();
						tkernel.computeTest
							(NA[Aidx], NB[Bidx], R,  C.wordSize(), C.array(),
							 NBG[BGidx], devTestResult,
							 bitGroups[BGidx].bitSize(), pos[BGidx], -1, b);
						recordLbf (testResult[NA[Aidx]+b], BGidx);
						}
//System.out.println(); System.out.flush();
					}
				else
					{
					for (int b = 0; b < NB[Bidx]; ++ b)
						zeroLbf (testResult[NA[Aidx]+b], BGidx);
					}

				// Compute odds ratios aggregated over all A and B inputs and
				// all bit groups. Find largest round exhibiting nonrandom
				// behavior.
				int maxnr = -1;
				int NINPUT = NA[Aidx] + NB[Bidx];
				for (int r = 0; r < R; ++ r)
					{
					double sumlbf = 0.0;
					for (int i = 0; i < NINPUT; ++ i)
						for (int g = 0; g < NBG[BGidx]; ++ g)
							sumlbf += testResult[i][r][g].aggLBF;
					if (sumlbf < 0.0)
						maxnr = Math.max (maxnr, r);
					}

				// Print results.
				long t4 = System.currentTimeMillis();
				NRR = Math.max (NRR, maxnr + 1);
				System.out.printf ("%d\t%d\t%d\t%d\t%d\t%d%n",
					Aidx, Bidx, BGidx, maxnr + 1, NRR, t4 - t3);
				System.out.flush();
				t3 = t4;
				}

			// Go to next combination of A input series, B input series, and bit
			// group.
			++ BGidx;
			if (BGidx == bitGroups.length)
				{
				BGidx = 0;
				++ Bidx;
				if (Bidx == B.length)
					{
					Bidx = 0;
					++ Aidx;
					}
				evaluate = true;
				}
			}

		long t2 = System.currentTimeMillis();
		System.out.printf ("%nNonrandom rounds = %d%n", NRR);
		System.out.printf ("Randomness margin = %.5f%n",
			1.0 - (double)NRR/(double)R);

		System.out.printf ("%nFinished %s%n", new Date (t2));
		System.out.printf ("%d msec%n", t2 - t1);
		}

	/**
	 * Download log Bayes factors from GPU and record in given matrix.
	 */
	private void recordLbf
		(TestResult[][] tr,
		 int BGidx)
		{
		devTestResult.devToHost();
		for (int i = 0; i < R; ++ i)
			for (int j = 0; j < NBG[BGidx]; ++ j)
				tr[i][j].copy (devTestResult.item[i][j]);
		}

	/**
	 * Zeroize log Bayes factors in given matrix.
	 */
	private void zeroLbf
		(TestResult[][] tr,
		 int BGidx)
		{
		for (int i = 0; i < R; ++ i)
			for (int j = 0; j < NBG[BGidx]; ++ j)
				tr[i][j].clear();
		}

	/**
	 * Print an error message and exit.
	 */
	private static void error
		(String msg)
		{
		System.err.printf ("AnalyzeSweep: %s%n", msg);
		usage();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 AnalyzeSweep [-nt=<NT>] \"<function>\" <A_file> <B_file> \"<test>\" <bitGroup_file> [<A_start> <B_start> <BG_start> <NRR>]");
		System.err.println ("<NT> = Number of GPU threads per block (default 768)");
		System.err.println ("<function> = Cryptographic function constructor expression");
		System.err.println ("<A_file> = File containing A input generator constructor expressions");
		System.err.println ("<B_file> = File containing B input generator constructor expressions");
		System.err.println ("<test> = Odds ratio test object constructor expression");
		System.err.println ("<bitGroup_file> = File containing bit group object constructor expressions");
		System.err.println ("<A_start> = Starting index for A input series (default 0)");
		System.err.println ("<B_start> = Starting index for B input series (default 0)");
		System.err.println ("<BG_start> = Starting index for bit groups (default 0)");
		System.err.println ("<NRR> = Maximum number of nonrandom rounds so far (default 0)");
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
