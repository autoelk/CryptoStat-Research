//******************************************************************************
//
// File:    Nola911.java
// Package: edu.rit.pjmr.example
// Unit:    Class edu.rit.pjmr.example.Nola911
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

package edu.rit.pjmr.example;

import edu.rit.draw.Drawing;
import edu.rit.draw.item.*;
import edu.rit.pj2.vbl.IntVbl;
import edu.rit.pjmr.Combiner;
import edu.rit.pjmr.Customizer;
import edu.rit.pjmr.Mapper;
import edu.rit.pjmr.PjmrJob;
import edu.rit.pjmr.Reducer;
import edu.rit.pjmr.TextDirectorySource;
import edu.rit.pjmr.TextId;
import edu.rit.util.Action;
import edu.rit.util.Pair;
import java.awt.Font;
import java.io.File;
import java.io.IOException;
import java.util.Date;

/**
 * Class Nola911 is the main program for a PJMR map-reduce job that analyzes
 * 911 call records from the City of New Orleans's Orleans Parish Communication
 * District (OPCD). The data files are available at
 * <A HREF="http://www.nola.gov/nopd/crime-data/911-calls-for-service/">http://www.nola.gov/nopd/crime-data/911-calls-for-service/</A>.
 * <P>
 * Usage: <TT>java pj2 [threads=<I>NT</I>] edu.rit.pjmr.example.Nola911
 * <I>nodes</I> <I>directory</I> <I>callType</I> <I>title</I>
 * <I>plotfile</I></TT>
 * <P>
 * The <I>nodes</I> argument is a comma-separated list of backend node names.
 * The program runs a separate mapper task on each of the given nodes. Each
 * mapper task has one source and <I>NT</I> mappers (default: one mapper). The
 * source reads all files in the given <I>directory</I> on the node where the
 * mapper task is running.
 * <P>
 * Each record (line) of each file consists of comma-separated fields; the first
 * record gives the field names. For each record with the given <I>callType</I>
 * (the second field), the program extracts the geographic location (latitude
 * and longitude, the last field) and counts how many calls occurred in each
 * 0.01-degree-square block (roughly, one-kilometer-square block). The program
 * creates a graphic depicting the counts, with the given <I>title</I>, and
 * stores the graphic in the given <I>plotfile</I>. Use the {@linkplain View
 * View} program to view the graphic.
 *
 * @author  Alan Kaminsky
 * @version 14-Jun-2016
 */
public class Nola911
	extends PjmrJob<TextId,String,Nola911Block,IntVbl>
	{

	/**
	 * PJMR job main program.
	 *
	 * @param  args  Command line arguments.
	 */
	public void main
		(String[] args)
		{
		// Parse command line arguments.
		if (args.length != 5) usage();
		String[] nodes = args[0].split (",");
		String directory = args[1];
		String callType = args[2];
		String title = args[3];
		String plotfile = args[4];

		// Determine number of mapper threads.
		int NT = Math.max (threads(), 1);

		// Print provenance.
		System.out.printf
			("$ java pj2 threads=%d edu.rit.pjmr.example.Nola911", NT);
		for (String arg : args)
			System.out.printf (" %s", arg);
		System.out.println();
		System.out.printf ("%s%n", new Date());
		System.out.printf ("Lat\tLon\tCount%n");
		System.out.flush();

		// Configure mapper tasks.
		for (String node : nodes)
			mapperTask (node)
				.source (new TextDirectorySource (directory))
				.mapper (NT, MyMapper.class, callType);

		// Configure reducer task.
		reducerTask()
			.runInJobProcess()
			.customizer (MyCustomizer.class)
			.reducer (MyReducer.class, title, plotfile);

		startJob();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 [threads=<NT>] edu.rit.pjmr.example.Nola911 <nodes> <directory> <callType> <title> <plotfile>");
		terminate (1);
		}

	/**
	 * Mapper class.
	 */
	private static class MyMapper
		extends Mapper<TextId,String,Nola911Block,IntVbl>
		{
		private static final IntVbl ONE = new IntVbl.Sum (1);
		private String callType;

		// Record call type.
		public void start
			(String[] args,
			 Combiner<Nola911Block,IntVbl> combiner)
			{
			this.callType = args[0];
			}

		// Process one data record.
		public void map
			(TextId id,
			 String data,
			 Combiner<Nola911Block,IntVbl> combiner)
			{
			int a = data.indexOf (',', 0);
			if (a == -1) return;
			int b = data.indexOf (',', a + 1);
			if (b == -1) return;
			if (! data.substring (a + 1, b) .equals (callType)) return;
			try
				{
				Nola911Block block = new Nola911Block (data);
				if (block.latitude() == 0.0 || block.longitude() == 0.0) return;
				combiner.add (block, ONE);
				}
			catch (Throwable exc)
				{
				}
			}
		}

	/**
	 * Customizer class for reducer task.
	 */
	private static class MyCustomizer
		extends Customizer<Nola911Block,IntVbl>
		{
		int minLat = Integer.MAX_VALUE;
		int minLon = Integer.MAX_VALUE;
		int maxLat = Integer.MIN_VALUE;
		int maxLon = Integer.MIN_VALUE;
		int maxCount = Integer.MIN_VALUE;

		public void start
			(String[] args,
			 Combiner<Nola911Block,IntVbl> combiner)
			{
			combiner.forEachItemDo (new Action<Pair<Nola911Block,IntVbl>>()
				{
				public void run (Pair<Nola911Block,IntVbl> pair)
					{
					minLat = Math.min (minLat, pair.key().lat());
					minLon = Math.min (minLon, pair.key().lon());
					maxLat = Math.max (maxLat, pair.key().lat());
					maxLon = Math.max (maxLon, pair.key().lon());
					maxCount = Math.max (maxCount, pair.value().intValue());
					}
				});
			}
		}

	/**
	 * Reducer class.
	 */
	private static class MyReducer
		extends Reducer<Nola911Block,IntVbl>
		{
		private static final int B = 12;
		String title;
		File plotfile;
		int minLat;
		int minLon;
		int maxLat;
		int maxLon;
		int maxCount;
		double logMaxCount;
		int W;
		int H;

		// Initialize.
		public void start (String[] args)
			{
			title = args[0];
			plotfile = new File (args[1]);
			MyCustomizer c = (MyCustomizer) customizer();
			minLat = c.minLat;
			minLon = c.minLon;
			maxLat = c.maxLat;
			maxLon = c.maxLon;
			maxCount = c.maxCount;
			logMaxCount = Math.log (c.maxCount);
			W = B*(maxLon - minLon + 1);
			H = B*(maxLat - minLat + 1);
			}

		// Process one 911 block.
		public void reduce
			(Nola911Block key,
			 IntVbl value)
			{
			System.out.printf ("%.2f\t%.2f\t%s%n",
				key.latitude(), key.longitude(), value);
			System.out.flush();
			new Rectangle() .width (B) .height (B)
				.sw (B*(key.lon() - minLon), -B*(key.lat() - minLat))
				.fill (new ColorFill()
					.hsb (hueFor (value.intValue()), 1.0f, 1.0f))
				.outline (null) .add();
			}

		// Returns the hue for the given count.
		private float hueFor
			(int count)
			{
			return (float)((logMaxCount - Math.log(count))/(3*logMaxCount));
			}

		// Store graphic in plotfile.
		public void finish()
			{
			System.out.printf ("Max\t\t%d%n", maxCount);
			Text.defaultFont (new Font (Font.SANS_SERIF, Font.PLAIN, 10));
			OutlinedItem.defaultOutline (new SolidOutline() .width (0.5f));
			Rectangle r = new Rectangle() .width (W) .height (H) .sw (0, 0)
				.fill (null) .add();
			new Text() .text (title) .s (r.n().n(B/2))
				.font (new Font (Font.SANS_SERIF, Font.PLAIN, 14)) .add();
			for (int i = minLon; i <= maxLon; ++ i)
				if ((i % 10) == 0)
					{
					Line l = new Line() .to (B*(i - minLon), 0) .vby (-H)
						.add();
					new Text() .text (String.format ("%.1f", i/100.0))
						.n (l.s().s(2)) .add();
					}
			for (int i = minLat; i <= maxLat; ++ i)
				if ((i % 10) == 0)
					{
					Line l = new Line() .to (0, -B*(i - minLat)) .hby (W)
						.add();
					new Text() .text (String.format ("%.1f", i/100.0))
						.e (l.w().w(4)) .add();
					}
			for (int i = 0; i <= 120; ++ i)
				new Rectangle() .width (B) .height (1) .sw (W+B, -i)
					.fill (new ColorFill()
						.hsb ((120.0f - i)/360.0f, 1.0f, 1.0f))
					.outline (null) .add();
			for (int i = 1; i < maxCount; i *= 10)
				new Text() .text (String.format ("%d", i))
					.w (W+2*B+4, -120*Math.log(i)/logMaxCount) .add();
			new Text() .text (String.format ("%d", maxCount))
				.sw (W+2*B+4, -121) .add();
			try
				{
				Drawing.write (plotfile);
				}
			catch (IOException exc)
				{
				exc.printStackTrace (System.err);
				}
			}
		}

	}
