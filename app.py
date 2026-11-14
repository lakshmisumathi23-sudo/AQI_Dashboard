            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#EAF4FF"))
        ]))

        story.append(table)

        story.append(Spacer(1, 8 * mm))

        # AQI category table

        story.append(
            Paragraph(
                "AQI Category Distribution",
                styles["Heading2"]
            )
        )

        category_pdf = [
            ["AQI Category", "Records", "Percentage"]
        ]

        for _, row in category_report.iterrows():

            category_pdf.append([
                str(row["AQI Category"]),
                str(int(row["Number of Records"])),
                f"{row['Percentage']:.2f}%"
            ])

        category_table = Table(
            category_pdf,
            colWidths=[60 * mm, 45 * mm, 45 * mm],
            repeatRows=1
        )

        category_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1976D2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6)
        ]))

        story.append(category_table)

        story.append(Spacer(1, 8 * mm))

        # State-wise AQI table

        story.append(
            Paragraph(
                "State-wise Average AQI",
                styles["Heading2"]
            )
        )

        state_pdf = [[
            "State",
            "Average",
            "Maximum",
            "Minimum"
        ]]

        for _, row in state_report.iterrows():

            state_pdf.append([
                str(row["State"]),
                f"{row['Average AQI']:.2f}",
                f"{row['Maximum AQI']:.0f}",
                f"{row['Minimum AQI']:.0f}"
            ])

        state_table = Table(
            state_pdf,
            colWidths=[65 * mm, 30 * mm, 30 * mm, 30 * mm],
            repeatRows=1
        )

        state_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1976D2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("FONTSIZE", (0, 0), (-1, -1), 8)
        ]))

        story.append(state_table)

        story.append(Spacer(1, 8 * mm))

        # Report note

        story.append(
            Paragraph(
                "Note: This report summarizes the historical "
                "records available in the uploaded dataset. "
                "It is not an official real-time AQI report.",
                styles["Italic"]
            )
        )

        document.build(story)

        buffer.seek(0)

        return buffer.getvalue()

    pdf_file = create_pdf_report()

    st.download_button(
        label="Download AQI Report (PDF)",
        data=pdf_file,
        file_name="Indian_AQI_Report.pdf",
        mime="application/pdf"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style="
    text-align: center;
    color: #000000;
    padding: 10px;
    font-weight: bold;">
    Indian AQI Dashboard | CPCB AQI Calculator and XGBoost Prediction
