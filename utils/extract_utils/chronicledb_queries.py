query_stddev = """
        SELECT TS, TE,
        AVG_LEAD_I, AVG_LEAD_II, AVG_LEAD_III,
        AVG_LEAD_AVR, AVG_LEAD_AVL, AVG_LEAD_AVF,
        AVG_LEAD_V1, AVG_LEAD_V2, AVG_LEAD_V3,
        AVG_LEAD_V4, AVG_LEAD_V5, AVG_LEAD_V6,
      ( STDDEV_LEAD_I + STDDEV_LEAD_II + STDDEV_LEAD_III
      + STDDEV_LEAD_AVR + STDDEV_LEAD_AVL + STDDEV_LEAD_AVF
      + STDDEV_LEAD_V1 + STDDEV_LEAD_V2 + STDDEV_LEAD_V3
      + STDDEV_LEAD_V4 + STDDEV_LEAD_V5 + STDDEV_LEAD_V6 ) AS DEV_SUM
    FROM (
      SELECT
        MIN(TIMESTEP) AS TS, MAX(TIMESTEP) AS TE,
        AVG(LEAD_I),   STDDEV(LEAD_I),
        AVG(LEAD_II),  STDDEV(LEAD_II),
        AVG(LEAD_III), STDDEV(LEAD_III),
        AVG(LEAD_AVR), STDDEV(LEAD_AVR),
        AVG(LEAD_AVL), STDDEV(LEAD_AVL),
        AVG(LEAD_AVF), STDDEV(LEAD_AVF),
        AVG(LEAD_V1),  STDDEV(LEAD_V1),
        AVG(LEAD_V2),  STDDEV(LEAD_V2),
        AVG(LEAD_V3),  STDDEV(LEAD_V3),
        AVG(LEAD_V4),  STDDEV(LEAD_V4),
        AVG(LEAD_V5),  STDDEV(LEAD_V5),
        AVG(LEAD_V6),  STDDEV(LEAD_V6)
      FROM {}
      WINDOW(COUNT 125 EVENTS JUMP 1 EVENT)
    ) INNER
    WHERE (TE - TS >= 124)
    """

query_subtract_stddev = """
                        SELECT timestep,
                            (lead_I - ({})) as LEAD_I,
                            (lead_II - ({})) as LEAD_II,
                            (lead_III - ({})) as LEAD_III,
                            (lead_AVR - ({})) as LEAD_AVR,
                            (lead_AVL - ({})) as LEAD_AVL,
                            (lead_AVF - ({})) as LEAD_AVF,
                            (lead_V1 - ({})) as LEAD_V1,
                            (lead_V2 - ({})) as LEAD_V2,
                            (lead_V3 - ({})) as LEAD_V3,
                            (lead_V4 - ({})) as LEAD_V4,
                            (lead_V5 - ({})) as LEAD_V5,
                            (lead_V6 - ({})) as LEAD_V6
                    FROM {}
                    """

query_rr_intervall = """
                SELECT r_r_start, r_r_end, r_r_distance
                FROM (
                    SELECT r_time_1 AS r_r_start, r_time_2 AS r_r_end, r_time_2 - r_time_1 AS r_r_distance
                    FROM (
                        SELECT r_time
                        FROM {}
                        MATCH_RECOGNIZE(
                            MEASURES
                                B.{} AS q_value,
                                E.timestep AS r_time,
                                E.{} AS r_value
                            PATTERN A B C+ D+ E
                            DEFINE
                                A AS TRUE,
                                B AS PREV({}) >= {},
                                C AS PREV({}) < {},
                                D AS (q_value+{}) < {} AND PREV({}) < {},
                                E AS (q_value+{}) < {} AND PREV({}) > {}
                            WITHIN 50 MILLISECONDS )
                        ) AS Peaks
                    MATCH_RECOGNIZE(
                        MEASURES
                            A.r_time AS r_time_1,
                            B.r_time AS r_time_2
                        PATTERN
                        A B DEFINE A AS TRUE, B AS TRUE )
                   ) AS Distances
                WINDOW (COUNT 5000 EVENTS JUMP 5000 EVENT)
                """

query_get_all_leads = """
                   SELECT
                   lead_I,lead_II,lead_III,
                   lead_AVR,lead_AVL,lead_AVF,
                   lead_V1,lead_V2,lead_V3,
                   lead_V4,lead_V5,lead_V6
                   FROM {}
                       WINDOW(COUNT 1 EVENTS JUMP 1 EVENTS)
               """

