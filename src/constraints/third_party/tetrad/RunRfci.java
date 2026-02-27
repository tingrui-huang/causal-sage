import edu.cmu.tetrad.data.BoxDataSet;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.data.VerticalIntDataBox;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Endpoint;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.search.Rfci;
import edu.cmu.tetrad.search.test.IndTestChiSquare;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Minimal RFCI runner for this repo.
 *
 * Reads a comma-separated CSV of integer-coded discrete variables (header required),
 * runs Tetrad RFCI with Chi-square test, and writes an edges CSV compatible with
 * Neuro-Symbolic-Reasoning/modules/prior_builder.py:
 *   source,target,edge_type,status
 *
 * Usage:
 *   java -cp <tetrad-lib-shaded.jar>;<compiled_classes_dir> RunRfci <input_csv> <output_csv> <alpha> <depth> <max_disc_path_len> <max_rows> <verbose>
 *
 * Notes:
 * - For bidirected edges (<->), we emit BOTH directions so downstream skeleton code can allow both.
 * - For undirected/partial/tail-tail, downstream code already treats as bidirectional.
 */
public final class RunRfci {
    private RunRfci() {}

    public static void main(String[] args) throws Exception {
        if (args.length < 7) {
            System.err.println("Usage: RunRfci <input_csv> <output_csv> <alpha> <depth> <max_disc_path_len> <max_rows> <verbose>");
            System.exit(2);
        }

        final Path input = Path.of(args[0]);
        final Path output = Path.of(args[1]);
        final double alpha = Double.parseDouble(args[2]);
        final int depth = Integer.parseInt(args[3]);               // -1 means unlimited in Tetrad
        final int maxDiscPathLen = Integer.parseInt(args[4]);      // -1 means unlimited in Tetrad
        final int maxRows = Integer.parseInt(args[5]);             // -1 means unlimited
        final boolean verbose = Boolean.parseBoolean(args[6]);

        DataAndVars dav = readDiscreteCsvAsBoxDataSet(input, maxRows);
        BoxDataSet dataSet = dav.dataSet;

        IndTestChiSquare test = new IndTestChiSquare(dataSet, alpha);
        test.setVerbose(verbose);

        Rfci rfci = new Rfci(test);
        rfci.setVerbose(verbose);
        rfci.setDepth(depth);
        rfci.setMaxDiscriminatingPathLength(maxDiscPathLen);

        Graph g = rfci.search();

        writeEdgesCsv(g, output);
    }

    private static final class DataAndVars {
        final BoxDataSet dataSet;
        DataAndVars(BoxDataSet ds) { this.dataSet = ds; }
    }

    private static DataAndVars readDiscreteCsvAsBoxDataSet(Path csvPath, int maxRows) throws IOException {
        try (BufferedReader br = Files.newBufferedReader(csvPath, StandardCharsets.UTF_8)) {
            String header = br.readLine();
            if (header == null) throw new IOException("Empty CSV: " + csvPath);

            String[] names = header.split(",", -1);
            int m = names.length;
            if (m <= 0) throw new IOException("No columns in CSV header: " + csvPath);

            int cap = 1024;
            int[][] cols = new int[m][cap];
            int[] maxVal = new int[m];
            Arrays.fill(maxVal, -1);
            int n = 0;

            String line;
            int[] row = new int[m];
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) continue;
                if (maxRows > 0 && n >= maxRows) break;

                if (n == cap) {
                    cap = cap * 2;
                    for (int j = 0; j < m; j++) {
                        cols[j] = Arrays.copyOf(cols[j], cap);
                    }
                }

                parseCsvIntRow(line, m, row);
                for (int j = 0; j < m; j++) {
                    int v = row[j];
                    cols[j][n] = v;
                    if (v > maxVal[j]) maxVal[j] = v;
                }
                n++;
            }

            // Trim columns to actual row count.
            for (int j = 0; j < m; j++) {
                cols[j] = Arrays.copyOf(cols[j], n);
            }

            List<Node> vars = new ArrayList<>(m);
            for (int j = 0; j < m; j++) {
                int k = maxVal[j] + 1;
                if (k <= 0) k = 1; // degenerate but safe
                vars.add(new DiscreteVariable(names[j], k));
            }

            VerticalIntDataBox box = new VerticalIntDataBox(cols);
            BoxDataSet ds = new BoxDataSet(box, vars);
            return new DataAndVars(ds);
        }
    }

    /**
     * Parse a comma-separated row of ints into out[m].
     * Assumes: no quotes, no escapes, all values are integer literals (e.g., 0,1,2).
     */
    private static void parseCsvIntRow(String line, int m, int[] out) throws IOException {
        int idx = 0;
        int col = 0;
        int len = line.length();

        while (col < m) {
            if (idx >= len) {
                throw new IOException("Row has fewer columns than header (expected " + m + "): " + line);
            }

            int sign = 1;
            char c = line.charAt(idx);
            if (c == '-') {
                sign = -1;
                idx++;
            }

            int val = 0;
            boolean hasDigit = false;
            while (idx < len) {
                c = line.charAt(idx);
                if (c >= '0' && c <= '9') {
                    hasDigit = true;
                    val = val * 10 + (c - '0');
                    idx++;
                } else {
                    break;
                }
            }
            if (!hasDigit) {
                throw new IOException("Non-integer cell at col " + col + ": " + line);
            }
            out[col] = val * sign;
            col++;

            // Skip delimiter if present
            if (col < m) {
                if (idx >= len || line.charAt(idx) != ',') {
                    throw new IOException("Row delimiter mismatch at col " + col + ": " + line);
                }
                idx++; // skip comma
            }
        }
    }

    private static void writeEdgesCsv(Graph g, Path outPath) throws IOException {
        Files.createDirectories(outPath.getParent());
        try (BufferedWriter bw = Files.newBufferedWriter(outPath, StandardCharsets.UTF_8)) {
            bw.write("source,target,edge_type,status\n");
            for (Edge e : g.getEdges()) {
                Node n1 = e.getNode1();
                Node n2 = e.getNode2();
                Endpoint ep1 = e.getEndpoint1();
                Endpoint ep2 = e.getEndpoint2();

                String a = n1.getName();
                String b = n2.getName();

                // Map endpoints to repo edge_type taxonomy.
                if (ep1 == Endpoint.TAIL && ep2 == Endpoint.ARROW) {
                    bw.write(a + "," + b + ",directed,accepted\n");
                } else if (ep1 == Endpoint.ARROW && ep2 == Endpoint.TAIL) {
                    bw.write(b + "," + a + ",directed,accepted\n");
                } else if (ep1 == Endpoint.ARROW && ep2 == Endpoint.ARROW) {
                    // bidirected: emit both directions
                    bw.write(a + "," + b + ",bidirected,accepted\n");
                    bw.write(b + "," + a + ",bidirected,accepted\n");
                } else if (ep1 == Endpoint.CIRCLE && ep2 == Endpoint.CIRCLE) {
                    bw.write(a + "," + b + ",undirected,accepted\n");
                } else if (ep1 == Endpoint.TAIL && ep2 == Endpoint.TAIL) {
                    bw.write(a + "," + b + ",tail-tail,accepted\n");
                } else if (ep1 == Endpoint.CIRCLE && ep2 == Endpoint.ARROW) {
                    bw.write(a + "," + b + ",partial,accepted\n");
                } else if (ep1 == Endpoint.ARROW && ep2 == Endpoint.CIRCLE) {
                    bw.write(b + "," + a + ",partial,accepted\n");
                } else {
                    // Fallback: treat as partial (most permissive for downstream).
                    bw.write(a + "," + b + ",partial,accepted\n");
                }
            }
        }
    }
}

