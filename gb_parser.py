from Bio import SeqIO
from pathlib import Path


def get_sorted(feature):

    if hasattr(feature.location, "parts"):
        parts = list(feature.location.parts)
    else:
        parts = [feature.location]
    parts.sort(key=lambda x: int(x.start))
    return parts


def parse_gb_to_gtf(input_path, output_path):

    with open(output_path, "w") as out:

        counter = 1 #

        for record in SeqIO.parse(input_path, "genbank"):
            genome = record.seq
            for feature in record.features:
                if feature.type != "CDS":
                    continue

                gene = feature.qualifiers.get("gene", [f"gene_{counter}"])[0]
                transcript = feature.qualifiers.get("protein_id", [f"prot_{counter}"])[0]

                gene_id = f"gene-{gene}"
                transcript_id = f"rna-{transcript}"
                strand = "+" if feature.location.strand == 1 else "-"

                parts = get_sorted(feature)
                n = len(parts)

                lengths = []

                for p in parts:
                    lengths.append(int(p.end) - int(p.start))

                for i, part in enumerate(parts):

                    start = int(part.start) + 1
                    end = int(part.end)

                    if strand == "+":

                        if i == 0:
                            cds_type = "Initial"
                        elif i == n - 1:
                            cds_type = "Terminal"
                        else:
                            cds_type = "Internal"
                        prev_len = sum(lengths[:i])

                    else:

                        if i == 0:
                            cds_type = "Terminal"
                        elif i == n - 1:
                            cds_type = "Initial"
                        else:
                            cds_type = "Internal"
                        prev_len = sum(lengths[i + 1:])

                    phase = (3 - (prev_len % 3)) % 3

                    out.write(
                        f"{record.id}\tGnomon\tCDS\t"
                        f"{start}\t{end}\t.\t{strand}\t{phase}\t"
                        f"gene_id \"{gene_id}\"; "
                        f"transcript_id \"{transcript_id}\"; "
                        f"cds_type \"{cds_type}\"; "
                        f"count \"{i+1}_{n}\";\n"
                    )



                if n == 1:

                    intron_start = int(parts[0].end) + 1
                    intron_end = intron_start + 9

                    out.write(
                        f"{record.id}\tGnomon\tintron\t"
                        f"{intron_start}\t{intron_end}\t.\t{strand}\t0\t"
                        f"gene_id \"{gene_id}\"; "
                        f"transcript_id \"{transcript_id}\"; "
                        f"count \"1_1\"; "
                        f"site_seq \"GT_AG\";\n"
                    )

                else:

                    for i in range(n - 1):

                        intron_start = int(parts[i].end) + 1
                        intron_end = int(parts[i + 1].start)

                        out.write(
                            f"{record.id}\tGnomon\tintron\t"
                            f"{intron_start}\t{intron_end}\t.\t{strand}\t0\t"
                            f"gene_id \"{gene_id}\"; "
                            f"transcript_id \"{transcript_id}\"; "
                            f"count \"{i+1}_{n-1}\"; "
                            f"site_seq \"GT_AG\";\n"
                        )

                if strand == "+":

                    start_pos = int(parts[0].start) + 1
                    stop_pos = int(parts[-1].end)

                    start_seq = genome[start_pos - 1:start_pos + 2]
                    stop_seq = genome[stop_pos - 3:stop_pos]

                    out.write(
                        f"{record.id}\tGnomon\tstart_codon\t"
                        f"{start_pos}\t{start_pos + 2}\t.\t{strand}\t0\t"
                        f"gene_id \"{gene_id}\"; "
                        f"transcript_id \"{transcript_id}\"; "
                        f"count \"1_1\"; "
                        f"site_seq \"{start_seq}\";\n"
                    )

                    out.write(
                        f"{record.id}\tGnomon\tstop_codon\t"
                        f"{stop_pos - 2}\t{stop_pos}\t.\t{strand}\t0\t"
                        f"gene_id \"{gene_id}\"; "
                        f"transcript_id \"{transcript_id}\"; "
                        f"count \"1_1\"; "
                        f"site_seq \"{stop_seq}\";\n"
                    )

                else:

                    start_pos = int(parts[-1].end)
                    stop_pos = int(parts[0].start) + 1

                    start_seq = genome[start_pos - 3:start_pos].reverse_complement()
                    stop_seq = genome[stop_pos - 1:stop_pos + 2].reverse_complement()

                    out.write(
                        f"{record.id}\tGnomon\tstart_codon\t"
                        f"{start_pos - 2}\t{start_pos}\t.\t{strand}\t0\t"
                        f"gene_id \"{gene_id}\"; "
                        f"transcript_id \"{transcript_id}\"; "
                        f"count \"1_1\"; "
                        f"site_seq \"{start_seq}\";\n"
                    )

                    out.write(
                        f"{record.id}\tGnomon\tstop_codon\t"
                        f"{stop_pos}\t{stop_pos + 2}\t.\t{strand}\t0\t"
                        f"gene_id \"{gene_id}\"; "
                        f"transcript_id \"{transcript_id}\"; "
                        f"count \"1_1\"; "
                        f"site_seq \"{stop_seq}\";\n"
                    )

                counter += 1


input_file = Path("data/annotation/AR9_annotation.gb")
output_file = Path("data/annotation_tiberius/AR9.gtf")

parse_gb_to_gtf(input_file, output_file)