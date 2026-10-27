package app.signsetu

import app.signsetu.ui.components.voiced
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NaturalVoiceTest {

    @Test
    fun theNaturalRecordingPlaysWhenChosenAndShipped() {
        val shipped = setOf("audio/hi/s1.ogg", "audio-natural/hi/s1.ogg")
        assertEquals("audio-natural/hi/s1.ogg", voiced("audio/hi/s1.ogg", natural = true, has = shipped::contains))
        assertEquals("audio/hi/s1.ogg", voiced("audio/hi/s1.ogg", natural = false, has = shipped::contains))
    }

    @Test
    fun aLanguageWithoutANaturalRecordingKeepsParler() {
        assertEquals("audio/as/s1.ogg", voiced("audio/as/s1.ogg", natural = true, has = { false }))
    }

    @Test
    fun eitherVoiceFillsInWhereTheOtherHasNoRecording() {
        val onlyNatural = setOf("audio-natural/hi/s5.ogg")
        assertEquals("audio-natural/hi/s5.ogg", voiced("audio/hi/s5.ogg", natural = false, has = onlyNatural::contains))
    }

    @Test
    fun aQuestionWithOnlyANaturalRecordingStillHasAudio() {
        assertEquals("audio/hi/s5.ogg", shipped.questionAudio("s5", "hi"))
        assertTrue(shipped.files.count { it.startsWith("audio-natural/") } > 900)
    }
}
